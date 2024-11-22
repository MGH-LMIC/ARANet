import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from natsort import natsorted


class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = (image_dir,)
        self.mask_dir = (mask_dir,)
        self.transform = transform

        images = []
        for root, dirs, files in os.walk(image_dir, topdown=False):
            for name in files:
                images.append(os.path.join(root, name))

        masks = []
        for root, dirs, files in os.walk(mask_dir, topdown=False):
            for name in files:
                masks.append(os.path.join(root, name))
        self.images = natsorted(images)
        self.masks = natsorted(masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
