# dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import re

class IGPairDataset(Dataset):
    def __init__(self, root, image_size=512):
        self.root = root
        self.cloth_dir = os.path.join(root, 'clothes')
        self.image_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'body_mask')
        self.pose_dir = os.path.join(root, 'openpose')
        self.densepose_dir = os.path.join(root, 'densepose')

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

        self.samples = []
        for fname in sorted(os.listdir(self.cloth_dir)):
            match = re.match(r'cloth_(\d{6})\.jpg', fname)
            if not match:
                continue
            cloth_id = match.group(1)

            # 找出所有對應的 image_xxxxxx_nn.jpg
            for f in sorted(os.listdir(self.image_dir)):
                if f.startswith(f'image_{cloth_id}_') and f.endswith('.jpg'):
                    nn_match = re.match(r'image_\d{6}_(\d{2})\.jpg', f)
                    if not nn_match:
                        continue
                    nn = nn_match.group(1)

                    # 各資料路徑
                    cloth_path = os.path.join(self.cloth_dir, f'cloth_{cloth_id}.jpg')
                    image_path = os.path.join(self.image_dir, f'image_{cloth_id}_{nn}.jpg')
                    mask_path = os.path.join(self.mask_dir, f'image_{cloth_id}_{nn}.png')
                    pose_path = os.path.join(self.pose_dir, f'image_{cloth_id}_{nn}.jpg')
                    dense_path = os.path.join(self.densepose_dir, f'image_{cloth_id}_{nn}.jpg')

                    # 檢查檔案存在性
                    missing = []
                    for path, name in zip(
                        [cloth_path, image_path, mask_path, pose_path, dense_path],
                        ['cloth', 'image', 'body_mask', 'openpose', 'densepose']
                    ):
                        if not os.path.exists(path):
                            missing.append(name)

                    if missing:
                        print(f"[Warning] Missing files for cloth {cloth_id}_{nn}: {', '.join(missing)}. Skipping.")
                        continue

                    self.samples.append((cloth_id, nn))

        print(f"[Dataset] Loaded {len(self.samples)} valid sample pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cloth_id, nn = self.samples[idx]

        cloth_path = os.path.join(self.cloth_dir, f'cloth_{cloth_id}.jpg')
        image_path = os.path.join(self.image_dir, f'image_{cloth_id}_{nn}.jpg')
        mask_path = os.path.join(self.mask_dir, f'image_{cloth_id}_{nn}.png')
        pose_path = os.path.join(self.pose_dir, f'image_{cloth_id}_{nn}.jpg')
        dense_path = os.path.join(self.densepose_dir, f'image_{cloth_id}_{nn}.jpg')

        cloth = self.transform(Image.open(cloth_path).convert('RGB'))
        image = self.transform(Image.open(image_path).convert('RGB'))
        mask = self.mask_transform(Image.open(mask_path).convert('L'))
        pose = self.transform(Image.open(pose_path).convert('RGB'))
        dense = self.transform(Image.open(dense_path).convert('RGB'))

        return {
            'cloth': cloth,
            'image': image,
            'mask': mask,
            'pose': pose,
            'densepose': dense
        }
