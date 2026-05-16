"""
Time-Series Mixer (TSMixer) for rainfall data processing.

This module implements an all-MLP architecture for processing univariate or
multivariate time-series data. It uses separate MLPs applied across the time
and feature dimensions to effectively learn temporal patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Scaler(nn.Module):
    """
    Scale activations dynamically based on the HeteroFL model slice rate.
    
    Ensures that variance remains consistent across sub-networks of different 
    capacities during distributed training.
    """
    def __init__(self, rate): 
        super().__init__()
        self.rate = rate
    def forward(self, x):
        return x / self.rate if self.training else x

class HeteroMixerLayer(nn.Module):
    """
    A single TSMixer layer containing time-mixing and feature-mixing MLPs.

    It sequentially mixes information along the time sequence length and then
    along the channel/feature dimension, supported by Batch Normalization.
    """
    def __init__(self, seq_len, num_features, rate):
        super().__init__()
        scaled_feat = num_features

        self.time_norm = nn.BatchNorm1d(scaled_feat, track_running_stats=False)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Linear(seq_len, seq_len)
        )

        self.feat_norm = nn.BatchNorm1d(scaled_feat, track_running_stats=False)
        self.feat_mlp = nn.Sequential(
            nn.Linear(scaled_feat, scaled_feat),
            nn.ReLU(),
            nn.Linear(scaled_feat, scaled_feat)
        )

        self.scaler = Scaler(rate)

    def forward(self, x):
        x_norm = self.time_norm(x)
        x = x + self.scaler(self.time_mlp(x_norm))

        x_norm = self.feat_norm(x)
        x_perm = x_norm.permute(0, 2, 1)
        x_mix = self.feat_mlp(x_perm).permute(0, 2, 1)
        return x + self.scaler(x_mix)

class RainTSMixer(nn.Module):
    """
    Rainfall Time-Series Encoder.

    Projects rainfall data into a higher-dimensional space and processes it
    through stacked HeteroMixer layers to extract a compact 1D feature vector
    for multimodal fusion.
    """
    def __init__(self, input_features, seq_len, out_dim, rate):
        super().__init__()
        scaled_in = int(input_features * rate) if input_features > 1 else 1
        hidden = int(32 * rate)

        self.proj = nn.Linear(scaled_in, hidden)
        self.l1 = HeteroMixerLayer(seq_len, hidden, rate)
        self.l2 = HeteroMixerLayer(seq_len, hidden, rate)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden, int(out_dim * rate))
        self.scaler = Scaler(rate)

    def forward(self, rain):
        x = self.proj(rain)
        x = x.permute(0, 2, 1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.gap(x).squeeze(-1)
        return self.scaler(self.head(x))
