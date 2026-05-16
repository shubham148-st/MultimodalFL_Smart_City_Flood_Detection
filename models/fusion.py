import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import DoubleConv, SpatialAttention
from .tsmixer import RainTSMixer
from .narx import WaterNARX


class TriModalFloodNet(nn.Module):
    """
    HeteroFL Tri-Modal Flood Detection Network.

    Modalities:
      - Satellite imagery  → CNN + Spatial Attention encoder (img_feat_dim)
      - Rainfall time-series → TSMixer encoder           (ts_feat_dim)
      - Water-level NARX  → NARX encoder                 (ts_feat_dim)

    The classifier receives the concatenated fusion vector after MC Dropout.
    When a modality is absent (img_only / sensor_only), its branch is zeroed out
    so the same fusion classifier handles all client types.
    """

    # Branch output sizes for rate=1.0
    IMG_DIM = 64   # CNN → AdaptiveAvgPool → 64-d
    TS_DIM  = 16   # TSMixer & NARX output

    def __init__(self, rate=1.0):
        super().__init__()
        self.rate = rate

        img_dim = int(self.IMG_DIM * rate)
        ts_dim  = int(self.TS_DIM  * rate)

        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove avgpool and fc to get spatial feature maps (512 channels)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze the pre-trained feature extractor EXCEPT layer4
        # Unfreezing layer4 allows the model to fine-tune high-level semantic features for satellite imagery
        for name, param in self.cnn.named_parameters():
            if '7' not in name: # index 7 is layer4
                param.requires_grad = False
            else:
                param.requires_grad = True
            
        self.spatial_attn = SpatialAttention()
        self.global_pool  = nn.AdaptiveAvgPool2d((1, 1))
        # ResNet-18 outputs 512 channels. We don't slice the input of this projection in HeteroFL
        # because the feature extractor is fixed and shared across all clients.
        self.img_proj     = nn.Linear(512, img_dim)

        # ── Rainfall TSMixer Branch ───────────────────────────────────────
        self.rain_net = RainTSMixer(1, 10, self.TS_DIM, rate)

        # ── Water-Level NARX Branch ───────────────────────────────────────
        self.water_net  = WaterNARX(20, 64, rate)
        self.water_proj = nn.Linear(1, ts_dim)

        # ── Fusion Head (Deep Stochastic) ────────────────────────────────
        # Two-layer classifier with dual dropout for meaningful MC variance.
        # Previous single-dropout head (p=0.3) produced near-zero MC variance
        # because stochasticity was too shallow; reviewers flagged this as
        # overconfident uncertainty estimates.
        fusion_dim = img_dim + ts_dim + ts_dim
        fusion_hid = fusion_dim // 2  # scales linearly with rate for HeteroFL slicing
        self.mc_dropout1 = nn.Dropout(p=0.5)   # ↑ from 0.3 for real variance
        self.fc_hidden   = nn.Linear(fusion_dim, fusion_hid)
        self.mc_dropout2 = nn.Dropout(p=0.4)   # second stochastic layer
        self.classifier  = nn.Linear(fusion_hid, 1)

        # Learnable temperature for post-hoc probability calibration.
        # Initialised to 1.0 (no-op); optimised during training so that
        # sigmoid(logit / T) produces well-calibrated confidence scores.
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, img, rain, water):
        # Determine batch & device from whichever input is present
        if img is not None:
            batch, device = img.size(0), img.device
        elif rain is not None:
            batch, device = rain.size(0), rain.device
        elif water is not None:
            batch, device = water.size(0), water.device
        else:
            raise ValueError("All inputs are None")

        img_dim = int(self.IMG_DIM * self.rate)
        ts_dim  = int(self.TS_DIM  * self.rate)

        # ── Image branch ──
        if img is not None:
            # ResNet expects 3 channels and standard 224x224 size
            img_3ch = img.repeat(1, 3, 1, 1)
            img_224 = F.interpolate(img_3ch, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Mathematical mapping to ImageNet domain:
            # 1. Min-max scale per patch to [0, 1]
            b, c, h, w = img_224.shape
            v_min = img_224.view(b, c, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
            v_max = img_224.view(b, c, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
            img_norm = (img_224 - v_min) / (v_max - v_min + 1e-8)
            
            # 2. Standard ImageNet normalization
            import torchvision.transforms as T
            img_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_norm)
            
            x = self.cnn(img_norm)
            x = self.spatial_attn(x) * x      # channel-wise spatial attention
            x = self.global_pool(x).flatten(1)
            x_img = F.relu(self.img_proj(x))
        else:
            x_img = torch.zeros(batch, img_dim, device=device)

        # ── Rain branch ──
        x_rain = self.rain_net(rain) if rain is not None else \
                 torch.zeros(batch, ts_dim, device=device)

        # ── Water branch ──
        if water is not None and rain is not None:
            narx_out = self.water_net(water, rain)
            x_water  = F.relu(self.water_proj(narx_out))
        else:
            x_water = torch.zeros(batch, ts_dim, device=device)

        # ── Modality Dropout (to prevent Sensor Dominance) ──
        # Sensor data is "easier" to learn, causing the global fusion head to ignore the CNN.
        # By randomly dropping modalities during training, we force the classifier to use the image branch.
        if self.training and img is not None and rain is not None:
            rand_val = torch.rand(1).item()
            if rand_val < 0.50:
                # Drop sensors 50% of the time -> aggressively forces reliance on images
                x_rain  = torch.zeros_like(x_rain)
                x_water = torch.zeros_like(x_water)
            elif rand_val > 0.85:
                # Drop image 15% of the time -> forces reliance on sensors
                x_img = torch.zeros_like(x_img)

        # ── Deep Stochastic Fusion ──
        fused = torch.cat([x_img, x_rain, x_water], dim=1)
        fused = self.mc_dropout1(fused)
        fused = F.relu(self.fc_hidden(fused))
        fused = self.mc_dropout2(fused)
        logits = self.classifier(fused)

        # Temperature scaling: logit / T  (T is clamped to avoid division by ≈0)
        return logits / self.temperature.clamp(min=0.01)