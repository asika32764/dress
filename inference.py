# inference.py

import torch
import argparse
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms as T
import os
from model import OOTDiffusion
from utils import get_device


# ========= 轉換 ==========
image_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])


# ========= 自動產生簡易 mask ==========
def generate_mask(image: torch.Tensor) -> torch.Tensor:
    # image: [1, 3, H, W], RGB normalized to [-1, 1]
    # 回到 [0,1] 再取亮度進行簡易 threshold masking
    gray = image.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    mask = (gray > 0).float()  # dummy mask: non-zero pixel
    return mask


# ========= 前處理 ==========
def load_inputs(cloth_path, image_path, device):
    cloth = image_transform(Image.open(cloth_path).convert("RGB")).unsqueeze(0).to(device)
    person = image_transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    mask = generate_mask(person)
    masked_person = person * mask
    return cloth, masked_person


# ========= 主推論流程 ==========
@torch.no_grad()
def run_inference(cloth_path, image_path, output_path="output.png"):
    device = get_device()
    print(f"[Device] Using {device}")

    # === 模型載入 ===
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device).eval()
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
    model = OOTDiffusion(unet).to(device)
    model.eval()

    clip_img_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_txt_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # === 載入圖片 ===
    cloth, masked_person = load_inputs(cloth_path, image_path, device)

    # === 編碼 latent ===
    z_cloth = vae.encode(cloth).latent_dist.sample() * 0.18215
    z_person = vae.encode(masked_person).latent_dist.sample() * 0.18215

    # === 初始雜訊 latent ===
    zt = torch.randn_like(z_person)
    t = torch.full((1,), 50, dtype=torch.long).to(device)

    # === 條件輸入 ===
    clip_img_feat = clip_img_encoder(cloth).last_hidden_state
    text_input = clip_tokenizer(["upperbody"], return_tensors="pt", padding=True).to(device)
    clip_txt_feat = clip_txt_encoder(**text_input).last_hidden_state
    cond = torch.cat([clip_img_feat, clip_txt_feat], dim=1)

    # === 推論 ===
    pred_noise = model(zt, t, z_cloth, cond, training=False)
    z0 = zt - pred_noise

    # === 解碼圖像 ===
    recon = vae.decode(z0 / 0.18215).sample
    save_image((recon + 1) / 2, output_path)
    print(f"[✓] Saved try-on result to {output_path}")


# ========= CLI ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOTDiffusion Inference")
    parser.add_argument("--cloth", type=str, required=True, help="Path to cloth image")
    parser.add_argument("--human", type=str, required=True, help="Path to human image")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")

    args = parser.parse_args()
    run_inference(args.cloth, args.human, args.output)
