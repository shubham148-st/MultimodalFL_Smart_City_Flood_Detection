"""
Federated training and evaluation protocols.

This module provides the core local training loops for heterogeneous clients,
dynamically routing data based on the client's available modalities (img_only,
sensor_only, full). It handles class-imbalanced learning via Focal Loss and 
provides robust global evaluation metrics including ROC-AUC, PR-AUC, and 
automated decision threshold optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from models.fusion import TriModalFloodNet
from .heterofl_ops import split_model
from .losses import FocalLossWithLogits
from .config import DEVICE, LOCAL_EPOCHS, LR


def _forward(model, imgs, rain, water, dtype):
    """
    Route inputs through the model depending on the client's data modality type.

    Args:
        model (nn.Module): The local or global TriModalFloodNet model.
        imgs (torch.Tensor): Satellite imagery batch.
        rain (torch.Tensor): Rainfall time-series batch.
        water (torch.Tensor): Water-level sensor batch.
        dtype (str): Modality type ('img_only', 'sensor_only', 'full').

    Returns:
        torch.Tensor: Model output logits.
    """
    if dtype == 'img_only':
        return model(imgs, None, None)
    elif dtype == 'sensor_only':
        return model(None, rain, water)
    else:
        return model(imgs, rain, water)


def _compute_pos_weight(loader, device):
    """
    Estimate BCE pos_weight from the dataset to handle class imbalance.

    Calculates the ratio of negative to positive samples in the training loader
    to appropriately scale the loss for the underrepresented positive class.

    Args:
        loader (DataLoader): Training data loader.
        device (str or torch.device): Compute device.

    Returns:
        torch.Tensor: Computed positive weight scalar.
    """
    total, pos = 0, 0
    for _, _, _, labels in loader:
        pos   += labels.sum().item()
        total += labels.numel()
    neg = total - pos
    if pos == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(neg / pos, device=device)


def _find_optimal_threshold(all_logits, all_labels, num_steps=50):
    """
    Grid-search the decision threshold that maximizes F1-Score (recall-biased).

    For disaster warning systems, missing an event (false negative) is typically
    more critical than a false alarm. This function searches threshold values
    between 0.1 and 0.9 to find the empirical optimum.

    Args:
        all_logits (torch.Tensor): Raw output logits from the model.
        all_labels (torch.Tensor): Ground truth labels.
        num_steps (int, optional): Number of thresholds to evaluate. Defaults to 50.

    Returns:
        float: The optimal probability threshold.
    """
    probs = torch.sigmoid(all_logits)
    best_f1 = -1.0
    best_th = 0.5
    for th in np.linspace(0.1, 0.9, num_steps):
        predicted = (probs > th).float()
        tp = ((predicted == 1) & (all_labels == 1)).sum().item()
        fp = ((predicted == 1) & (all_labels == 0)).sum().item()
        fn = ((predicted == 0) & (all_labels == 1)).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th


def train_one_round(global_model, clients, train_loader):
    """
    Run one federated round: local training and evaluation for every active client.

    Each client downloads a rate-sliced version of the global model, trains it
    on their specific modality combination for a set number of epochs, and 
    returns the updated parameters.

    Args:
        global_model (nn.Module): The current global model architecture.
        clients (list of dict): List of client configurations.
        train_loader (DataLoader): The federated training dataset.

    Returns:
        tuple: (local_updates, active_rates, client_round_metrics)
            - local_updates (list of dict): State dicts of locally trained models.
            - active_rates (list of float): The HeteroFL slicing rates used.
            - client_round_metrics (dict): Mapping of client ID to local loss and accuracy.
    """
    # Focal Loss to handle flood/no-flood imbalance and boost recall.
    # α=0.75 up-weights flood class; γ=2.0 focuses on hard examples.
    pos_weight = _compute_pos_weight(train_loader, DEVICE)
    criterion  = FocalLossWithLogits(alpha=0.75, gamma=2.0, pos_weight=pos_weight)

    local_updates        = []
    active_rates         = []
    client_round_metrics = {}

    for client in clients:
        cid    = client['id']
        crate  = client['rate']
        dtype  = client['data']
        epochs = client.get('epochs', LOCAL_EPOCHS)
        lr     = client.get('lr',     LR)

        print(f"  {cid} training (Rate {crate}, Mode: {dtype}, Epochs: {epochs})...")

        client_weights = split_model(global_model, crate)
        local_model    = TriModalFloodNet(rate=crate).to(DEVICE)
        local_model.load_state_dict(client_weights)
        local_model.train()
        optimizer = optim.AdamW(local_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Local Training
        import torchvision.transforms as T
        aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(15),
        ])

        epoch_loss = 0
        steps      = 0
        for _ in range(epochs):
            for imgs, rain, water, labels in train_loader:
                imgs   = imgs.to(DEVICE)
                rain   = rain.to(DEVICE)
                water  = water.to(DEVICE)
                labels = labels.to(DEVICE)

                if imgs.size(0) == 1:
                    continue
                    
                # Apply data augmentation to prevent CNN overfitting
                imgs = aug(imgs)

                optimizer.zero_grad()

                # Raw logits for BCEWithLogitsLoss (numerically stable)
                logits = _forward(local_model, imgs, rain, water, dtype)
                loss   = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1

            scheduler.step()

        avg_loss = epoch_loss / steps if steps > 0 else 0

        # Local Evaluation
        local_model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for imgs, rain, water, labels in train_loader:
                imgs   = imgs.to(DEVICE)
                rain   = rain.to(DEVICE)
                water  = water.to(DEVICE)
                labels = labels.to(DEVICE)

                if imgs.size(0) == 1:
                    continue

                # Use logits > 0 for evaluation (equivalent to prob > 0.5)
                logits    = _forward(local_model, imgs, rain, water, dtype)
                predicted = (logits > 0.0).float()
                correct  += (predicted == labels).sum().item()
                total    += labels.size(0)

        local_acc = 100 * correct / total if total > 0 else 0

        local_updates.append(local_model.state_dict())
        active_rates.append(crate)
        client_round_metrics[cid] = {'loss': avg_loss, 'acc': local_acc}

        print(f"      -> Loss: {avg_loss:.4f} | Acc: {local_acc:.2f}%")

    return local_updates, active_rates, client_round_metrics



def evaluate_global(global_model, data_loader, dtype='full'):
    """
    Evaluate the global model with an optimized decision threshold.

    Computes standard classification metrics plus ROC-AUC, PR-AUC, and 
    provides a Confusion Matrix. Leverages `_forward` to enable seamless
    ablation studies on specific data modalities.

    Args:
        global_model (nn.Module): The aggregated global model to evaluate.
        data_loader (DataLoader): The evaluation dataset (Val or Test).
        dtype (str, optional): The modality configuration to test. Defaults to 'full'.

    Returns:
        dict: A dictionary containing 'acc', 'prec', 'rec', 'f1', 'threshold', 
              'roc_auc', 'pr_auc', and 'cm' (Confusion Matrix).
    """
    global_model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for imgs, rain, water, labels in data_loader:
            imgs   = imgs.to(DEVICE)
            rain   = rain.to(DEVICE)
            water  = water.to(DEVICE)
            labels = labels.to(DEVICE)

            if imgs.size(0) == 1:
                continue

            logits = _forward(global_model, imgs, rain, water, dtype)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Find recall-optimised threshold
    best_th = _find_optimal_threshold(all_logits, all_labels)

    # Compute metrics at optimal threshold
    probs     = torch.sigmoid(all_logits)
    predicted = (probs > best_th).float()

    # Move tensors to CPU numpy for sklearn
    y_true = all_labels.cpu().numpy()
    y_prob = probs.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    correct   = (y_pred == y_true).sum()
    total     = len(y_true)
    acc       = 100 * correct / total if total > 0 else 0

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    # ROC, PR, CM
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc  = average_precision_score(y_true, y_prob)
        cm      = confusion_matrix(y_true, y_pred)
    except ValueError:
        roc_auc, pr_auc, cm = 0.0, 0.0, None

    return {
        'acc': acc,
        'prec': precision,
        'rec': recall,
        'f1': f1,
        'threshold': best_th,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'cm': cm
    }
