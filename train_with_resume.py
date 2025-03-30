# train.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel
from torchvision.transforms import transforms
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPProcessor
from dataset import IGPairDataset
from model import OOTDiffusion
from utils import get_device
from tqdm import tqdm
import os
import argparse
import re

def train():
    # ==== 裝置 ====
    device = get_device()
    print(f"[Device] Using {device}")

    # ==== 預訓練模型載入 ====
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device).eval()
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
    clip_img_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_txt_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    model = OOTDiffusion(unet).to(device)
    model.train()

    # ==== 資料集與 DataLoader ====
    dataset = IGPairDataset("./data", image_size=512)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 初始化繼續功能
    resume_path = None
    start_epoch = 0
    best_loss = float('inf')

    if resume_mode == 'latest':
        if os.path.exists("checkpoints"):
            ckpts = [f for f in os.listdir("checkpoints") if f.startswith("ootd_epoch_")]
            if ckpts:
                latest = max(ckpts, key=lambda x: int(re.findall(r'\\d+', x)[0]))
                resume_path = os.path.join("checkpoints", latest)
    elif resume_mode == 'best':
        resume_path = "checkpoints/best_model.pt"

    if resume_path and os.path.exists(resume_path):
        print(f"[Resume] Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float('inf'))
        print(f"[Resume] Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, 10):
        total_loss = 0
        count = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            cloth = batch["cloth"].to(device)
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            # ==== 轉 latent ====
            with torch.no_grad():
                z_cloth = vae.encode(cloth).latent_dist.sample() * 0.18215
                z_image = vae.encode(image * mask).latent_dist.sample() * 0.18215  # masked person image

            # ==== 噪聲與時間步 ====
            noise = torch.randn_like(z_image)
            t = torch.randint(0, 1000, (z_image.shape[0],), device=device).long()
            zt = z_image + noise

            # ==== Conditioning（CLIP image + text）====
            with torch.no_grad():
                # 轉到 CPU 做 resize
                clip_ready_cloth = torch.nn.functional.interpolate(
                    cloth.cpu(), size=(224, 224), mode='bilinear', align_corners=False
                ).to(device)  # 再轉回 GPU 或 MPS
                # clip_img_feat = clip_img_encoder(clip_ready_cloth).last_hidden_state
                # text_input = clip_tokenizer(["upperbody"] * zt.size(0), return_tensors="pt", padding=True).to(device)
                # clip_txt_feat = clip_txt_encoder(**text_input).last_hidden_state

                # cond = torch.cat([clip_img_feat, clip_txt_feat], dim=1)
                # cond = clip_img_feat

                text_input = clip_tokenizer(["upperbody"] * zt.size(0), return_tensors="pt", padding=True).to(device)
                cond = clip_txt_encoder(**text_input).last_hidden_state

            # ==== 前向與 loss ====
            pred_noise = model(zt, t, z_cloth, cond, training=True)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            print(f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / count
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")

        # 儲存 checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/ootd_epoch_{epoch:02d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, ckpt_path)
        print(f"[Checkpoint] Saved at {ckpt_path}")

        # 如果是目前最佳 loss，也儲存一份 best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = "checkpoints/best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, best_path)
            print(f"[✓] New best model saved at {best_path} with loss {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', choices=['latest', 'best'], default=None)
    args = parser.parse_args()
    resume_mode = args.resume

    train()
