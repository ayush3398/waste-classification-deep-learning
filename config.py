"""Central settings. Every script imports from here so the whole project stays consistent."""
import os, random
import numpy as np

SEED = 42            # fixed random seed -> reproducible split, weights init, shuffling
IMG_SIZE = 128       # every image is resized to 128x128
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 20          # maximum epochs (early stopping may stop sooner)
PATIENCE = 5         # stop if validation loss does not improve for 5 epochs
DROPOUT = 0.5

DATA_DIR = "data/dataset-resized"   # TrashNet (dataset-resized version)
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = "models"

# TrashNet has 6 folders. It has NO "organic/food" class, so the default is the
# 5 material classes below (the 137-image "trash" folder is left out).
# To use all 6 classes:  INCLUDE_TRASH=1 python train.py
ALL_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASSES = ALL_CLASSES if os.environ.get("INCLUDE_TRASH") == "1" else ALL_CLASSES[:5]
NUM_CLASSES = len(CLASSES)

# Normalisation: (pixel - 0.5) / 0.5  -> values in [-1, 1]. Fixed constants, so nothing is learnt from test data.
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

for d in (RESULTS_DIR, PLOTS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)


def set_seed(seed=SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
