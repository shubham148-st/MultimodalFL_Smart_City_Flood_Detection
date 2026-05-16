"""
Main execution script for HeteroFL Multimodal Flood Detection.

This module orchestrates the federated learning pipeline across heterogeneous clients:
- Loads the multimodal dataset (satellite imagery, river gauge, rainfall).
- Initializes the global TriModalFloodNet model.
- Executes federated training rounds, simulating clients with varying channel widths.
- Evaluates the aggregated global model on held-out validation and test sets.
- Performs ablation studies and Monte Carlo dropout uncertainty quantification.
"""

import torch
from torch.utils.data import DataLoader

from models.fusion import TriModalFloodNet
from heterofl.config import DEVICE, NUM_ROUNDS, BATCH_SIZE, CLIENTS
from heterofl.data_loader import get_dataset
from heterofl.heterofl_ops import aggregate_models
from heterofl.training import train_one_round, evaluate_global
from heterofl.mc_dropout import monte_carlo_uncertainty
from heterofl.plotting import save_plots


def main():
    """
    Executes the main federated learning pipeline.
    
    Workflow:
    1. Dataset loading and Train/Val/Test splitting.
    2. Initialization of the global model.
    3. Loop through federated rounds:
       - Local client training on HeteroFL-sliced subnetworks.
       - Model aggregation at the server.
       - Global validation and checkpointing.
    4. Final evaluation, modality ablation, and uncertainty quantification on the Test set.
    """
    print(f" Starting HeteroFL Multimodal Flood Detection on {DEVICE}")
    print(f" Rounds: {NUM_ROUNDS} | Clients: {[c['id'] for c in CLIENTS]}\n")

    # Data
    train_ds, val_ds, test_ds = get_dataset()
    if train_ds is None:
        print("Dataset is empty. Exiting.")
        return
        
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Global Model
    global_model = TriModalFloodNet(rate=1.0).to(DEVICE)
    n_params = sum(p.numel() for p in global_model.parameters())
    print(f" Global model parameters: {n_params:,}\n")

    # Metrics Storage
    global_history  = []   # (acc, precision, recall, f1) per round
    client_metrics  = {c['id']: {'loss': [], 'acc': []} for c in CLIENTS}
    best_f1         = -1.0
    best_round      = 0
    best_state      = None

    # Federated Rounds
    for round_idx in range(NUM_ROUNDS):
        print(f"\nGlobal Round {round_idx + 1}/{NUM_ROUNDS}")

        local_updates, active_rates, round_metrics = train_one_round(
            global_model, CLIENTS, train_loader
        )

        # Accumulate per-client metrics
        for cid, m in round_metrics.items():
            client_metrics[cid]['loss'].append(m['loss'])
            client_metrics[cid]['acc'].append(m['acc'])

        # Aggregate
        print("   Aggregating...")
        global_model = aggregate_models(global_model, local_updates, active_rates)

        # Global Evaluation on Validation Set (Acc + F1)
        metrics = evaluate_global(global_model, val_loader)
        acc, prec, rec, f1, best_th = metrics['acc'], metrics['prec'], metrics['rec'], metrics['f1'], metrics['threshold']
        
        global_history.append({'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'threshold': best_th})
        print(f"   Val Acc: {acc:.2f}% | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | Threshold: {best_th:.3f}")

        # Best-model checkpoint (federated aggregation can regress)
        if f1 > best_f1:
            best_f1    = f1
            best_round = round_idx + 1
            best_state = {k: v.clone() for k, v in global_model.state_dict().items()}
            print(f"   * New best model (Val F1={f1:.4f}) -- saved checkpoint")

    # Restore best model & save
    if best_state is not None:
        global_model.load_state_dict(best_state)
    torch.save(global_model.state_dict(), "flood_model.pth")
    print(f"\nBest model (Round {best_round}) saved to 'flood_model.pth'")

    # Re-evaluate the best model
    test_metrics = evaluate_global(global_model, test_loader)
    acc, prec, rec, f1, best_th = test_metrics['acc'], test_metrics['prec'], test_metrics['rec'], test_metrics['f1'], test_metrics['threshold']
    roc_auc, pr_auc, cm = test_metrics['roc_auc'], test_metrics['pr_auc'], test_metrics['cm']

    print("\n" + "="*55)
    print(f"  FINAL RESULTS ON TEST SET (Best Model - Round {best_round})")
    print("="*55)
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  Threshold : {best_th:.4f}")
    print(f"  Temp (T)  : {global_model.temperature.item():.4f}")
    print("="*55)
    if cm is not None:
        print("  Confusion Matrix:")
        print(f"    TN: {cm[0][0]:<5} | FP: {cm[0][1]:<5}")
        print(f"    FN: {cm[1][0]:<5} | TP: {cm[1][1]:<5}")
        print("="*55)

    # Ablation Study
    print("\n[ Ablation Study (Test Set) ]")
    for mode in ['img_only', 'sensor_only', 'full']:
        mode_metrics = evaluate_global(global_model, test_loader, dtype=mode)
        print(f" Mode: {mode:<12} | F1: {mode_metrics['f1']:.4f} | Acc: {mode_metrics['acc']:.2f}% | Recall: {mode_metrics['rec']:.4f}")

    # Monte Carlo Dropout Uncertainty
    monte_carlo_uncertainty(global_model, test_loader)

    # Graphs
    acc_history = [m['acc'] for m in global_history]
    save_plots(NUM_ROUNDS, acc_history, client_metrics, CLIENTS)


if __name__ == "__main__":
    main()