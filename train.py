# train.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel
from torchvision.transforms import transforms
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPProcessor
from dataset import IGPairDataset
from model import OOTDiffusion
from profiler import Profiler
from utils import get_device
from tqdm import tqdm
import os
import argparse
import re
from torch.amp import autocast, GradScaler

def train(options):
    profiler = Profiler(enabled=bool(options.profiler))
    profiler.reset()

    # ==== 裝置 ====
    device = get_device(options.device)
    print(f"[Device] Using {device}")

    # AMP
    scaler = GradScaler(device)

    # ==== 預訓練模型載入 ====
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device).eval()
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
    # clip_img_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_txt_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # 如果 clip encoder 支援 float16，你可以讓它用更少記憶體：
    clip_txt_encoder = clip_txt_encoder.half()

    model = OOTDiffusion(unet).to(device)
    model.train()

    # ==== 資料集與 DataLoader ====
    dataset = IGPairDataset("./data", image_size=512)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True, num_workers=options.workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 預先計算 Clip Tokenizer 輸入（避免每 batch 重複）
    base_text_input = clip_tokenizer(["upperbody"], return_tensors="pt", padding=True)

    # 初始化繼續功能
    resume_mode = options.resume
    resume_path = None
    start_epoch = 0
    best_loss = float('inf')
    profiler.log('>> init')
    if resume_mode == 'latest':
        if os.path.exists("output"):
            ckpts = [f for f in os.listdir("output") if f.startswith("dress_epoch_")]
            if ckpts:
                latest = max(ckpts, key=lambda x: int(re.findall(r'\\d+', x)[0]))
                resume_path = os.path.join("output", latest)
    elif resume_mode == 'best':
        resume_path = "output/best_model.pt"

    if resume_path and os.path.exists(resume_path):
        print(f"[Resume] Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float('inf'))
        print(f"[Resume] Resumed from epoch {start_epoch}")
    profiler.log('>> start epoch looping')
    for epoch in range(start_epoch, 10):
        total_loss = 0
        count = 0

        progressbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in progressbar:
            cloth = batch["cloth"].to(device)
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            profiler.reset()
            profiler.log('>> Batch loop')

            # ==== 轉 latent ====
            with torch.no_grad():
                profiler.log('>> z_cloth z_image')
                z_cloth = vae.encode(cloth).latent_dist.sample() * 0.18215
                z_image = vae.encode(image * mask).latent_dist.sample() * 0.18215  # masked person image

            profiler.log('>> noise time sync')
            # ==== 噪聲與時間步 ====
            noise = torch.randn_like(z_image)
            t = torch.randint(0, 1000, (z_image.shape[0],), device=device).long()
            zt = z_image + noise
            profiler.log('>> noice time sync end')

            # ==== Conditioning（CLIP image + text）====
            with torch.no_grad():
                profiler.log('>> CLIP image + text while')
                # 轉到 CPU 做 resize
                # clip_ready_cloth = torch.nn.functional.interpolate(
                #     cloth.cpu(), size=(224, 224), mode='bilinear', align_corners=False
                # )
                # clip_img_feat = clip_img_encoder(clip_ready_cloth).last_hidden_state
                # text_input = clip_tokenizer(["upperbody"] * zt.size(0), return_tensors="pt", padding=True).to(device)
                # clip_txt_feat = clip_txt_encoder(**text_input).last_hidden_state

                # cond = torch.cat([clip_img_feat, clip_txt_feat], dim=1)
                # cond = clip_img_feat

                text_input = {k: v.expand(zt.size(0), -1).to(device) for k, v in base_text_input.items()}
                profiler.log('>> text encode')
                cond = clip_txt_encoder(**text_input).last_hidden_state

            # ==== 前向與 loss ====
            # profiler.log('>> 前向與 loss')
            # pred_noise = model(zt, t, z_cloth, cond, training=True)
            # loss = F.mse_loss(pred_noise, noise)
            # profiler.log('>> optimier')
            # optimizer.zero_grad()
            # profiler.log('>> backward')
            # loss.backward()
            # profiler.log('>> step')
            # optimizer.step()
            # profiler.log('>> optimier end')

            profiler.log('>> zero grad')
            optimizer.zero_grad()

            profiler.log('>> autocast')
            with autocast(device.type):
                pred_noise = model(zt, t, z_cloth, cond, training=True)
                loss = F.mse_loss(pred_noise, noise)

            profiler.log('>> backward')
            scaler.scale(loss).backward()

            profiler.log('>> step optimizer')
            scaler.step(optimizer)

            profiler.log('>> scalar.update')
            scaler.update()

            total_loss += loss.item()
            count += 1
            progressbar.set_postfix(loss = f"{loss.item():.4f}")

        avg_loss = total_loss / count
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")

        # 儲存 checkpoint
        os.makedirs("output", exist_ok=True)
        ckpt_path = f"output/dress_epoch_{epoch:02d}.pt"
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
            best_path = "output/best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, best_path)
            print(f"[✓] New best model saved at {best_path} with loss {best_loss:.4f}")

def print_options(options):
    print("\n🛠️  Training Configuration:")
    print("-" * 40)
    for key, value in vars(options).items():
        print(f"{key:15s}: {value}")
    print("-" * 40 + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resume', choices=['latest', 'best'], default='latest')
    parser.add_argument('--profiler', type=int, default=0, help="Enable step-wise profiler (1 to enable)")
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--workers', type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print_options(args)

    train(args)
