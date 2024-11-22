# read whole case
from natsort import natsorted
import matplotlib.pyplot as plt
from PIL import Image
import glob
import shutil
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from natsort import natsorted
import numpy as np
from scipy import ndimage
import skimage
from skimage import measure
import cv2
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
val_transform = A.Compose(
    [
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)


def read_case(path):
    files = natsorted(glob.glob(f"{path}/*.png"))
    images = []
    for i in files:
        img = Image.open(i).convert("RGB")
        images.append(img)
    return images


def check_accuracy(path, model, device=DEVICE, transform=val_transform):
    image = Image.open(path).convert("RGB")
    image = np.array(image)
    augmentations = transform(image=image)
    x = augmentations["image"]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = preds.argmax(dim=1).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

    return preds


def save_preds(preds, dirs, idx):
    img = Image.fromarray(preds)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    img.save(f"{dirs}/{idx}.png")


def find_smallest_square(n: int) -> int:
    i = 1
    while i**2 < n:
        i += 1
    return i


def remove_small_patches(mask, min_size):
    # Label the different patches in the mask
    labeled_mask, num_labels = ndimage.label(mask)

    # Get the size of each patch
    sizes = ndimage.sum(mask, labeled_mask, range(num_labels + 1))

    # Create a mask to keep large patches
    keep_mask = sizes >= min_size
    keep_mask[0] = 0  # Ensure background (label 0) is not removed

    # Apply the mask to keep large patches and remove small ones
    cleaned_mask = keep_mask[labeled_mask].astype(mask.dtype)

    return cleaned_mask


def filter_minors(image):
    labels_mask = measure.label(image)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    # print(regions)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
    labels_mask[labels_mask != 0] = 1
    mask = labels_mask

    labels_mask = measure.label(mask)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=False)
    # print(regions)
    if len(regions) > 1:
        for rg in regions[:3]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 1
    # labels_mask[labels_mask!=0] = 255
    mask = labels_mask
    # strel = np.ones((10, 10))
    # dilated = binary_dilation(mask, structure=strel)
    return np.uint(mask)


def fill_minor_parts(mask, min_size):
    # Convert the mask to an 8-bit unsigned integer type
    mask = mask.astype(np.uint8)

    # Use connectedComponentsWithStats to get the size of each component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    for i in range(1, num_labels):
        # If the size of the component is less than or equal to 10
        if stats[i, cv2.CC_STAT_AREA] <= min_size:
            # Set the corresponding pixels in the mask to zero
            mask[labels == i] = 0

    return mask
