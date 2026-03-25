import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, images_dir, masks_dir, extra_dirs=None, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.extra_dirs = extra_dirs or []
        self.transform = transform

        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

        print(len(self.images), len(self.masks))
        assert len(self.images) == len(self.masks), \
            "Number of images and masks must match"

        # optional extra tiles that should align with images/masks
        self.extras = []
        for extra_dir in self.extra_dirs:
            extra_files = sorted(os.listdir(extra_dir))
            assert len(extra_files) == len(self.images), \
                f"Number of extra tiles in {extra_dir} must match number of images"
            self.extras.append((extra_dir, extra_files))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        extras = []
        for extra_dir, extra_files in self.extras:
            extra_path = os.path.join(extra_dir, extra_files[idx])
            extra_tile = Image.open(extra_path).convert("L")
            extras.append(extra_tile)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            extras = [self.transform(e) for e in extras]

        # optionally concatenate extra channels into image tensor
        if extras:
            image = torch.cat([image] + extras, dim=0)

        # ensure mask is binary (0 or 1)
        mask = (mask > 0).float()

        return image, mask