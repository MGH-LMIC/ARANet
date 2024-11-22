# import os
# import sys

# models_path = "./models"

# sys.path.append(os.path.abspath(models_path))

# import torch
# import torch.nn as nn
# from models.aranetfpn_aspp2 import ARANetFPN


# print("import successful")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# folder = "trained_models"
# models = {
#     "ARANetFPN": {
#         "class": ARANetFPN,
#         "checkpoint": f"./{folder}/<class 'aranetfpn_aspp2.ARANetFPN'>_256_split_0.pt",
#     }
# }


# def load_model(model_class, checkpoint_path, device):
#     model = torch.load(checkpoint_path, map_location=device)
#     return model.to(device)


# for model_name, model_info in models.items():
#     print(f"\nEvaluating {model_name}")
#     model = load_model(model_info["class"], model_info["checkpoint"], device)
#     print("model loaded successfully ...")


import os
from pathlib import Path


# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"


print("Contents of trained_models directory:")
print(os.listdir(TRAINED_MODELS_DIR))
