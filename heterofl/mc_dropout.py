import numpy as np
import torch
import multiprocessing

from .config import DEVICE, MC_ITERATIONS


def _plot_histogram(predictions, mean_pred, iterations, all_variances):
    """Generate uncertainty histogram in isolated process."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-pass flood probability for sample #0
    axes[0].hist(predictions, bins=25, color='steelblue', alpha=0.8, edgecolor='black')
    axes[0].axvline(mean_pred, color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean Prob: {mean_pred:.3f}')
    axes[0].set_title(f'MC Dropout Predictions — Sample #0 ({iterations} Passes)')
    axes[0].set_xlabel('Predicted Flood Probability')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Right: per-sample variance distribution (all evaluated samples)
    axes[1].hist(all_variances, bins=20, color='coral', alpha=0.8, edgecolor='black')
    axes[1].set_title('Per-Sample Predictive Variance Distribution')
    axes[1].set_xlabel('Variance (σ²)')
    axes[1].set_ylabel('Count')
    axes[1].axvline(float(np.mean(all_variances)), color='darkred', linestyle='dashed',
                    linewidth=2, label=f'Mean Var: {float(np.mean(all_variances)):.5f}')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig('results/uncertainty_histogram_figure.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def monte_carlo_uncertainty(model, dataloader, iterations=None, num_samples=5):
    """Run MC Dropout uncertainty quantification on multiple samples.

    Uses a FULL batch (not duplicated samples) so that BatchNorm1d with
    track_running_stats=False computes meaningful batch statistics.

    Parameters
    ----------
    model      : trained TriModalFloodNet
    dataloader : DataLoader with real data
    iterations : number of stochastic forward passes
    num_samples: how many samples to evaluate (from the first batch)

    Returns
    -------
    results : list of dict with keys 'true_label', 'mean_pred', 'variance', 'correct'
    """
    if iterations is None:
        iterations = MC_ITERATIONS

    print("\n--- Monte Carlo Dropout Uncertainty Quantification ---")

    # Enable ALL dropout layers; keep BatchNorm in eval mode.
    # The deeper stochastic fusion head (mc_dropout1 + mc_dropout2) now
    # produces meaningful variance across MC forward passes.
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    # Use a full real batch (BatchNorm needs diverse samples for valid statistics)
    imgs, rain, water, labels = next(iter(dataloader))
    imgs   = imgs.to(DEVICE)
    rain   = rain.to(DEVICE)
    water  = water.to(DEVICE)
    labels = labels.to(DEVICE)
    batch_size = imgs.size(0)
    num_samples = min(num_samples, batch_size)

    all_predictions = [[] for _ in range(num_samples)]

    with torch.no_grad():
        for _ in range(iterations):
            # Forward the FULL batch each time (keeps BN happy)
            logits = model(imgs, rain, water)  # [B, 1]
            probs  = torch.sigmoid(logits)
            for s in range(num_samples):
                all_predictions[s].append(probs[s].item())

    results = []
    print(f"\n{'Sample':<8} {'True':>6} {'Mean%':>8} {'Var':>10} {'Verdict':>10}")
    print("-" * 48)

    for s in range(num_samples):
        preds     = all_predictions[s]
        true_lbl  = labels[s].item()
        mean_pred = float(np.mean(preds))
        variance  = float(np.var(preds))
        predicted = 'Flood' if mean_pred > 0.5 else 'No Flood'
        actual    = 'Flood' if true_lbl == 1.0 else 'No Flood'
        correct   = (predicted == actual)

        results.append({
            'true_label': true_lbl,
            'mean_pred':  mean_pred,
            'variance':   variance,
            'correct':    correct,
        })

        status = '[OK]' if correct else '[FAIL]'
        print(f"  #{s:<4}  {actual:>6}  {mean_pred*100:>7.2f}%  {variance:>9.5f}  {status:>10}")

    # Aggregate uncertainty statistics (journal table row)
    all_variances = [r['variance'] for r in results]
    print(f"\n  Uncertainty Summary:")
    print(f"    Mean Variance : {np.mean(all_variances):.6f}")
    print(f"    Max  Variance : {np.max(all_variances):.6f}")
    print(f"    Min  Variance : {np.min(all_variances):.6f}")
    print(f"    Std  Variance : {np.std(all_variances):.6f}")

    # Use first sample's predictions for the histogram
    main_preds = all_predictions[0]
    main_mean  = float(np.mean(main_preds))

    # Save raw predictions to CSV
    try:
        np.savetxt('csv_results/mc_predictions.csv', main_preds, delimiter=',',
                   header='flood_probability', comments='')
        print(f"\n Saved MC predictions to 'csv_results/mc_predictions.csv'")
    except Exception:
        pass

    # Generate histogram in subprocess
    try:
        p = multiprocessing.Process(target=_plot_histogram,
                                    args=(main_preds, main_mean, iterations,
                                          all_variances))
        p.start()
        p.join(timeout=15)
        if p.exitcode == 0:
            print(" Saved uncertainty plot to 'results/uncertainty_histogram_figure.png'")
        else:
            print(" [INFO] Histogram plot failed. Use mc_predictions.csv to plot externally.")
    except Exception:
        print(" [INFO] Histogram plot skipped.")

    return results
