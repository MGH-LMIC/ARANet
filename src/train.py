import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import importlib.util
from typing import Any, Dict, Type
from utils import *
from pathlib import Path
import sys

from models.archs import NestedUNet
from models.resunetplusplus import build_resunetplusplus
from models.deeplab import DeepLabWrapper
from models.doubleunet import build_doubleunet
from models.aranetfpn_aspp2 import ARANetFPN
from models.se_fpn import SEFPN

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

PROJECT_ROOT = SCRIPT_DIR.parent


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CLASSES = {
    "ARANetFPN": ARANetFPN,
    "build_doubleunet": build_doubleunet,
    "build_resunetplusplus": build_resunetplusplus,
    "DeepLabWrapper": DeepLabWrapper,
    "NestedUNet": NestedUNet,
    "SEFPN": SEFPN,
}


class ConfigLoader:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_active_model_config(self) -> Dict[str, Any]:
        """Get the configuration for the active model"""
        active_model = self.config["models"]["active_model"]
        print(active_model, self.config["models"][active_model])
        return self.config["models"][active_model]


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        # Move data to device without explicit float conversion
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
    return model


def main(resize, SPLIT):
    config_loader = ConfigLoader("config.yaml")
    config = config_loader.config
    model_config = config_loader.get_active_model_config()

    # Load model class
    try:
        model_class = MODEL_CLASSES[model_config["class_name"]]

        if model_config["params"] == "None":
            model = model_class().to(DEVICE)
            print(f"Successfully loaded model: {model_config['class_name']}")
        else:
            model = model_class(**model_config["params"]).to(DEVICE)
            print(f"Successfully loaded model: {model_config['class_name']}")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    try:
        data_path = Path(config["data"]["data_path"])
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    NUM_EPOCHS = 55
    NUM_WORKERS = 2
    PIN_MEMORY = True
    LOAD_MODEL = False
    TRAIN_IMG_DIR = PROJECT_ROOT / f"data/split_{SPLIT}/train_images/"
    TRAIN_MASK_DIR = PROJECT_ROOT / f"data/split_{SPLIT}/train_masks/"
    VAL_IMG_DIR = PROJECT_ROOT / f"data/split_{SPLIT}/val_images/"
    VAL_MASK_DIR = PROJECT_ROOT / f"data/split_{SPLIT}/val_masks/"
    IMAGE_HEIGHT = resize
    IMAGE_WIDTH = resize

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=50, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    filename = f"./{model_config['class_name']}_{resize}_split_{SPLIT}.pt"

    if LOAD_MODEL:
        load_checkpoint(torch.load(filename), model)

    scaler = torch.cuda.amp.GradScaler()

    dice_zero = 0
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch+1} /{NUM_EPOCHS}, size: {resize}, split: {SPLIT}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # check accuracy
        dice = check_accuracy(val_loader, model, device=DEVICE)

        # save model
        if dice > dice_zero:
            # save_checkpoint(checkpoint, filename)
            print(
                "=> Saving checkpoint ---------------------------------------------------------"
            )
            torch.save(model.state_dict(), filename)
            dice_zero = dice
            new_line = "\n"
            with open(f"./logs/log_{model_config['class_name']}.txt", "a") as f:
                f.write(
                    f"Split: {SPLIT}, Epoch: {epoch+1}, Dice score: {dice: 0.3f}{new_line}"
                )


if __name__ == "__main__":
    resizes = [256]
    splits = [0]

    for i in splits:
        for j in resizes:
            main(resize=j, SPLIT=i)
