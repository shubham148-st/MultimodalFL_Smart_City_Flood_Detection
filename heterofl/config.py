import torch

# === Device & Training Configuration ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ROUNDS    = 10    # more rounds → CNN has more iterations to converge
LOCAL_EPOCHS  = 5     # base epochs per round
BATCH_SIZE    = 16
LR            = 0.001
MC_ITERATIONS = 50    # Monte Carlo dropout forward passes


CLIENTS = [
    {'id': 'satellite_img', 'rate': 0.5, 'data': 'img_only',    'epochs': 8,  'lr': 0.0005},
    {'id': 'River_Gauge',   'rate': 0.5, 'data': 'sensor_only', 'epochs': 5,  'lr': 0.001},
    {'id': 'HQ_Server',     'rate': 1.0, 'data': 'full',        'epochs': 5,  'lr': 0.001},
]
