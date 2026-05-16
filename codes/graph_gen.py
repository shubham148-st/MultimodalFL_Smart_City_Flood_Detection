import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import — prevents segfault
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# All paths relative to project root (parent of codes/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Generating Journal Figures...")

# 1. Generate Uncertainty Histogram
try:
    mc_path = os.path.join(ROOT, 'mc_predictions.csv')
    df_mc = pd.read_csv(mc_path)
    preds = df_mc['flood_probability'].values
    mean_pred = np.mean(preds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(preds, bins=15, color='steelblue', alpha=0.8, edgecolor='black')
    ax.axvline(mean_pred, color='red', linestyle='dashed', linewidth=2, label=f'Mean Prob: {mean_pred:.3f}')
    ax.set_title('Prediction Uncertainty via Monte Carlo Dropout\n(50 Forward Passes)')
    ax.set_xlabel('Predicted Flood Probability')
    ax.set_ylabel('Frequency')
    ax.legend()
    out = os.path.join(ROOT, 'uncertainty_histogram_figure.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f" -> Saved {out}")
    plt.close(fig)
except FileNotFoundError:
    print(" -> mc_predictions.csv not found. Skipping histogram.")

# 2. Generate Global Accuracy Graph
try:
    global_path = os.path.join(ROOT, 'global_accuracy.csv')
    df_global = pd.read_csv(global_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_global['round'], df_global['accuracy'], marker='o', color='black', linewidth=2)
    ax.set_title('Federated Global Model Accuracy')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
    out = os.path.join(ROOT, 'global_accuracy_graph.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f" -> Saved {out}")
    plt.close(fig)
except FileNotFoundError:
    print(" -> global_accuracy.csv not found.")

# 3. Generate Per-Client Accuracy & Loss Graphs
try:
    colors = ['r', 'g', 'b']
    client_ids = ['satellite_img', 'River_Gauge', 'HQ_Server']

    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))

    for i, cid in enumerate(client_ids):
        csv_path = os.path.join(ROOT, f'client_{cid}_metrics.csv')
        df = pd.read_csv(csv_path)
        ax_acc.plot(df['round'], df['accuracy'], marker='s', linestyle='--',
                    color=colors[i], label=f'{cid}')
        ax_loss.plot(df['round'], df['loss'], marker='x', linestyle=':',
                     color=colors[i], label=f'{cid}')

    ax_acc.set_title('Local Client Accuracy per Round')
    ax_acc.set_xlabel('Round')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.legend()
    ax_acc.grid(True, linestyle='--', alpha=0.7)
    out = os.path.join(ROOT, 'client_accuracy_graph.png')
    fig_acc.savefig(out, dpi=300, bbox_inches='tight')
    print(f" -> Saved {out}")
    plt.close(fig_acc)

    ax_loss.set_title('Local Client Loss per Round')
    ax_loss.set_xlabel('Round')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.7)
    out = os.path.join(ROOT, 'client_loss_graph.png')
    fig_loss.savefig(out, dpi=300, bbox_inches='tight')
    print(f" -> Saved {out}")
    plt.close(fig_loss)

except FileNotFoundError as e:
    print(f" -> Client metrics CSV not found: {e}")

print("Done.")