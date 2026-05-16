"""
Focal Loss for binary classification with class imbalance.

In flood detection, the standard BCEWithLogitsLoss tends to saturate on
easy-to-classify 'no flood' samples, which dominate the dataset.
Focal Loss (Lin et al., ICCV 2017) down-weights well-classified examples
and concentrates learning signal on hard positives — floods that the model
currently misses — directly boosting recall.

Journal reference:
    'We adopt Focal Loss with γ=2.0 and class-aware α to mitigate the
    high false-negative rate inherent in imbalanced flood datasets,
    achieving X% recall improvement over standard BCE.'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLogits(nn.Module):
    """Binary Focal Loss operating on raw logits (numerically stable).

    Parameters
    ----------
    alpha : float
        Weighting factor for the positive class.  α > 0.5 biases toward
        recall (fewer missed floods).  Estimated from data if not provided.
    gamma : float
        Focusing parameter.  γ = 0 ≡ standard BCE.  γ = 2 is the
        standard recommendation from Lin et al.
    pos_weight : torch.Tensor or None
        If provided, used as the BCE pos_weight (class-frequency ratio).
        When `alpha` is also set, both effects combine: alpha re-scales
        the focal modulation while pos_weight handles class imbalance
        inside the base BCE term.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 pos_weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard BCE with logits (no reduction — we apply focal weighting)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=self.pos_weight
        )

        # Predicted probability
        p = torch.sigmoid(logits)
        # p_t = p when y=1, (1-p) when y=0
        p_t = targets * p + (1 - targets) * (1 - p)

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        return (focal_weight * bce).mean()
