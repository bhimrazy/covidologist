"""
This configuration file will store important variables and parameters used across our driver scripts. 
Instead of re-defining them in every script, we’ll simply define them once here (and thereby make our 
code cleaner and easier to read).
"""
import os
import pytz
import torch
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count

CPU_COUNT = cpu_count()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_FOLDER_PATH = os.path.join(BASE_DIR, "dataset")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ARTIFACTS_DIR = os.path.join(Path(__file__).resolve().parent, "artifacts")

PYTORCH_FILE_NAME = datetime.now(tz=pytz.timezone(
    'Asia/kathmandu')).strftime('%Y-%m-%d_%H:%M:%S_model.pth.tar')
ONNX_FILE_NAME = datetime.now(tz=pytz.timezone(
    'Asia/kathmandu')).strftime('%Y-%m-%d_%H:%M:%S_model.onnx')
LOGS_FILE_NAME = os.path.join(ARTIFACTS_DIR, "logs", datetime.now(tz=pytz.timezone(
    'Asia/kathmandu')).strftime('logs_%Y-%m-%d_%H:%M:%S.json'))
