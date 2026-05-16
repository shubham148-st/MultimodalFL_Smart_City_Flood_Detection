import numpy as np
import multiprocessing


def _do_plot(rounds, global_history, client_metrics_data, client_ids):
    """Run plotting in an isolated process to avoid matplotlib segfaults."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plot 1: Global Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, global_history, marker='o', linestyle='-',
            color='black', linewidth=2, label='Global Model')
    ax.set_title('Global Model Accuracy')
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    fig.savefig('results/global_accuracy_graph.png')
    plt.close(fig)

    # Plot 2: Per-Client Accuracy
    colors = ['r', 'g', 'b']
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cid in enumerate(client_ids):
        ax.plot(rounds, client_metrics_data[cid]['acc'], marker='s', linestyle='--',
                color=colors[i], label=f'{cid} Acc')
    ax.set_title('Local Client Accuracy per Round')
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.savefig('results/client_accuracy_graph.png')
    plt.close(fig)

    # Plot 3: Per-Client Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cid in enumerate(client_ids):
        ax.plot(rounds, client_metrics_data[cid]['loss'], marker='x', linestyle=':',
                color=colors[i], label=f'{cid} Loss')
    ax.set_title('Local Client Loss per Round')
    ax.set_xlabel('Round')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.savefig('results/client_loss_graph.png')
    plt.close(fig)


def save_plots(num_rounds, global_history, client_metrics, clients):
    """Generate and save all training-result graphs.

    Saves CSV metrics first, then tries matplotlib in a subprocess
    to avoid segfaults from crashing the main pipeline.
    """
    rounds = list(range(1, num_rounds + 1))
    client_ids = [c['id'] for c in clients]

    # Always save metrics as CSV (safe fallback)
    try:
        np.savetxt('csv_results/global_accuracy.csv',
                   np.column_stack([rounds, global_history]),
                   delimiter=',', header='round,accuracy', comments='')
        for cid in client_ids:
            np.savetxt(f'csv_results/client_{cid}_metrics.csv',
                       np.column_stack([rounds,
                                        client_metrics[cid]['acc'],
                                        client_metrics[cid]['loss']]),
                       delimiter=',', header='round,accuracy,loss', comments='')
        print("Metrics saved as CSV files.")
    except Exception as e:
        print(f"[WARNING] Could not save CSV metrics: {e}")

    # Attempt matplotlib in a child process (isolated from main process)
    try:
        # Prepare serializable data
        cm_data = {cid: {'acc': client_metrics[cid]['acc'],
                         'loss': client_metrics[cid]['loss']} for cid in client_ids}

        p = multiprocessing.Process(target=_do_plot,
                                    args=(rounds, global_history, cm_data, client_ids))
        p.start()
        p.join(timeout=30)

        if p.exitcode == 0:
            print("Graphs saved: results/global_accuracy_graph.png, results/client_accuracy_graph.png, results/client_loss_graph.png")
        else:
            print(f"[WARNING] Plot process exited with code {p.exitcode}. Use CSV files to plot externally.")
    except Exception as e:
        print(f"[WARNING] Could not generate plots: {e}")
        print("Use the saved CSV files to plot externally.")
    