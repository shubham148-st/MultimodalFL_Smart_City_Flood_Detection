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
    """Route inputs through the model depending on the client data type."""
    if dtype == 'img_only':
        return model(imgs, None, None)
    elif dtype == 'sensor_only':
        return model(None, rain, water)
    else:
        return model(imgs, rain, water)


def _compute_pos_weight(loader, device):
    """Estimate BCE pos_weight from the dataset to handle class imbalance."""
    total, pos = 0, 0
    for _, _, _, labels in loader:
        pos   += labels.sum().item()
        total += labels.numel()
    neg = total - pos
    if pos == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(neg / pos, device=device)


def _find_optimal_threshold(all_logits, all_labels, num_steps=50):
    """Grid-search the decision threshold that maximises F1 (recall-biased).

    For flood warning systems, missing a flood (false negative) is far
    worse than a false alarm.  We search thresholds from 0.1 to 0.9 and
    pick the one with best F1, which inherently raises recall when the
    optimum lies below the default 0.5.
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
    """Run one federated round: local training + evaluation for every client.

    Returns
    -------
    local_updates : list[state_dict]
    active_rates  : list[float]
    client_round_metrics : dict  {cid: {'loss': float, 'acc': float}}
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

        # --- Local Training ---
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

        # --- Local Evaluation ---
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
    """Evaluate the global model with optimised decision threshold.

    Computes standard metrics plus ROC-AUC, PR-AUC, and Confusion Matrix.
    Uses _forward to allow ablation studies on specific modalities.
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
