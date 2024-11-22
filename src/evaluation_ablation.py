import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, recall_score, precision_score
import matplotlib.pyplot as plt
import warnings
from dataset import MRIDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import os
import sys
from pathlib import Path
import io


# Setup paths
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

PROJECT_ROOT = SCRIPT_DIR.parent
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"


from models.aranetfpn_aspp2 import ARANetFPN
from models.doubleunet import build_doubleunet
from models.resunetplusplus import build_resunetplusplus
from models.deeplab import DeepLabWrapper
from models.archs import NestedUNet
from models.se_fpn import SEFPN


# Constants
SPLIT_NUM = 0
BATCH_SIZE = 4
NUM_WORKERS = 2
PIN_MEMORY = True
NUM_CLASSES = 3


TEST_IMG_DIR = PROJECT_ROOT / f"data/split_{SPLIT_NUM}/test_images"
TEST_MASK_DIR = PROJECT_ROOT / f"data/split_{SPLIT_NUM}/test_masks"


MODEL_CHECKPOINTS = {
    "ARANetFPN": "ARANetFPN_state_dict.pt",
    "DoubleUNet": "DoubleUNet_state_dict.pt",
    "ResUNet++": "ResUNet++_state_dict.pt",
    "DeepLabv3": "DeepLabv3_state_dict.pt",
    "UNet++": "UNet++_state_dict.pt",
    "SE_FPN": "SEFPN_256_split_0.pt",
}


def load_model(checkpoint_name, device):
    """Load model from state dict"""
    checkpoint_path = TRAINED_MODELS_DIR / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Create model instance
    if "ARANetFPN" in checkpoint_name:
        model = ARANetFPN(num_classes=3)
    elif "DoubleUNet" in checkpoint_name:
        model = build_doubleunet(num_classes=3)
    elif "ResUNet++" in checkpoint_name:
        model = build_resunetplusplus(num_classes=3)
    elif "DeepLabv3" in checkpoint_name:
        model = DeepLabWrapper(num_classes=3)
    elif "UNet++" in checkpoint_name:
        model = NestedUNet(num_classes=3)
    elif "SEFPN" in checkpoint_name:
        model = SEFPN(num_classes=3)

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    return model.to(device)


def multi_class_dice_coefficient(y_true, y_pred, num_classes):
    dice_scores = []
    for class_id in range(num_classes):
        y_true_class = (y_true == class_id).flatten()
        y_pred_class = (y_pred == class_id).flatten()
        intersection = np.sum(y_true_class * y_pred_class)
        smooth = 1e-7
        if np.sum(y_true_class) == 0 and np.sum(y_pred_class) == 0:
            dice_scores.append(
                1.0
            )  # Both true and pred are empty, consider it a perfect match
        else:
            dice = (2.0 * intersection + smooth) / (
                np.sum(y_true_class) + np.sum(y_pred_class) + smooth
            )
            dice_scores.append(dice)
    return np.mean(dice_scores)


def multi_class_iou(y_true, y_pred, num_classes):
    iou_scores = []
    for class_id in range(num_classes):
        y_true_class = (y_true == class_id).flatten()
        y_pred_class = (y_pred == class_id).flatten()
        intersection = np.sum(y_true_class * y_pred_class)
        union = np.sum(y_true_class) + np.sum(y_pred_class) - intersection
        smooth = 1e-7
        if union == 0:
            iou_scores.append(
                1.0
            )  # Both true and pred are empty, consider it a perfect match
        else:
            iou = (intersection + smooth) / (union + smooth)
            iou_scores.append(iou)
    return np.mean(iou_scores)


def safe_metric_calculation(metric_func, y_true, y_pred, **kwargs):
    """Safely calculate metrics with warning suppression"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return metric_func(y_true, y_pred, **kwargs)
        except:
            return np.nan


def evaluate_model(model, test_loader, device, num_classes):
    """Evaluate model with safe metric calculation"""
    model.eval()

    metrics = {
        "Dice Score": [],
        "IoU Score": [],
        "F1 Score": [],
        "Recall": [],
        "Precision": [],
    }

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            predicted = predicted.cpu().numpy()
            masks = masks.cpu().numpy()

            # Using safe metric calculation for all metrics
            metrics["Dice Score"].append(
                safe_metric_calculation(
                    multi_class_dice_coefficient,
                    masks,
                    predicted,
                    num_classes=num_classes,
                )
            )

            metrics["IoU Score"].append(
                safe_metric_calculation(
                    multi_class_iou, masks, predicted, num_classes=num_classes
                )
            )

            metrics["F1 Score"].append(
                safe_metric_calculation(
                    f1_score,
                    masks.flatten(),
                    predicted.flatten(),
                    average="macro",
                    labels=range(num_classes),
                    zero_division=1.0,
                )
            )

            metrics["Recall"].append(
                safe_metric_calculation(
                    recall_score,
                    masks.flatten(),
                    predicted.flatten(),
                    average="macro",
                    labels=range(num_classes),
                    zero_division=1.0,
                )
            )

            metrics["Precision"].append(
                safe_metric_calculation(
                    precision_score,
                    masks.flatten(),
                    predicted.flatten(),
                    average="macro",
                    labels=range(num_classes),
                    zero_division=1.0,
                )
            )

    # Calculate mean of metrics, ignoring NaN values
    return {k: np.nanmean(v) for k, v in metrics.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data transforms and loader
    val_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    # Verify data paths
    print(f"Testing data paths:")
    print(f"Images: {TEST_IMG_DIR} (exists: {TEST_IMG_DIR.exists()})")
    print(f"Masks: {TEST_MASK_DIR} (exists: {TEST_MASK_DIR.exists()})")

    test_ds = MRIDataset(
        image_dir=str(TEST_IMG_DIR),
        mask_dir=str(TEST_MASK_DIR),
        transform=val_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    results = {}

    # Evaluate each model
    for model_name, checkpoint_name in MODEL_CHECKPOINTS.items():
        print(f"\nEvaluating {model_name}")
        try:
            model = load_model(checkpoint_name, device)
            results[model_name] = evaluate_model(
                model, test_loader, device, NUM_CLASSES
            )
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue

    # Print and save results
    if results:
        # Print results
        for model_name, metrics in results.items():
            print(f"\nResults for {model_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

        # Plot results
        plot_results(results)

        # Save results
        df = pd.DataFrame(results).T
        results_path = PROJECT_ROOT / "results" / "model_evaluation_results.csv"
        results_path.parent.mkdir(exist_ok=True)
        df.to_csv(results_path)
        print(f"\nResults saved to {results_path}")
    else:
        print("No results to display - all models failed to evaluate")


def plot_results(results):
    """Plot comparison of model results"""
    metrics = list(next(iter(results.values())).keys())
    x = np.arange(len(results))
    width = 0.15

    fig, ax = plt.subplots(figsize=(15, 8))
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in results]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_ylabel("Scores")
    ax.set_title("Model Comparison")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(results.keys())
    ax.legend()
    plt.tight_layout()

    # Save plot
    plot_path = PROJECT_ROOT / "results" / "model_comparison.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    main()
