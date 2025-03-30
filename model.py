# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from diffusers import UNet2DConditionModel

# === Outfitting Dropout ===
def apply_outfitting_dropout(latent, dropout_prob=0.1, training=True):
    if training and torch.rand(1).item() < dropout_prob:
        return torch.zeros_like(latent)
    return latent

# === Outfitting Fusion (簡化模擬版) ===
def outfitting_fusion(human_feat, garment_feat):
    """
    簡化版的融合：直接將 garment features 與 human features 合併再裁剪
    """
    fused = torch.cat([human_feat, garment_feat], dim=-1)  # concat on width
    return fused[..., :human_feat.shape[-1]]  # crop 回原來的大小

# === 模型主體 ===
class OOTDiffusion(nn.Module):
    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.denoising_unet = unet
        self.outfitting_unet = copy.deepcopy(unet)
        self.guidance_scale = 1.5

        # 新增：投影服裝 feature 成 UNet conditioning 維度
        self.garment_proj = nn.Linear(4, 768)  # 假設通道數是 4，視 g_feat 輸出而定

    def forward(self, zt, t, garment_latent, conditioning, training=True):
        garment_latent = apply_outfitting_dropout(garment_latent, 0.1, training)

        with torch.no_grad():
            g_feat = self.outfitting_unet(
                sample=garment_latent, timestep=t, encoder_hidden_states=conditioning
            ).sample  # [B, C, H, W]

        # 攤平成每張圖一個特徵向量： [B, C]
        g_vector = g_feat.mean(dim=[2, 3])  # [B, C]

        # 投影成 conditioning 維度： [B, 768]
        g_projected = self.garment_proj(g_vector)  # [B, 768]

        # 擴展成 sequence： [B, 1, 768] -> broadcast 加到 conditioning
        g_broadcast = g_projected.unsqueeze(1)

        fused_conditioning = conditioning + g_broadcast  # [B, 77, 768]

        noise_pred = self.denoising_unet(zt, t, fused_conditioning).sample
        return noise_pred

