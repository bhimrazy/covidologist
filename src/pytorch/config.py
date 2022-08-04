"""
This configuration file will store important variables and parameters used across our driver scripts. 
Instead of re-defining them in every script, we’ll simply define them once here (and thereby make our 
code cleaner and easier to read).
"""
import os
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_FOLDER_PATH = os.path.join(BASE_DIR, "src/dataset")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
