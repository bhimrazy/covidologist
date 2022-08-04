"""
This script is  used to build our dataset directory, along with the train,val and test subdirectories.
"""

import os
import glob
import shutil
import random
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

CPU_COUNT = cpu_count()
print("The CPU count is:", CPU_COUNT)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_FOLDER_PATH = os.path.join(BASE_DIR, "temp")
DATASET_FOLDER_PATH = os.path.join(BASE_DIR, "src/dataset")

CLASSES = ['positive', 'negative']


def prepare_dataset_folders():
    """This function prepares the dataset folder ready to be used.
       It divides the folder as train,val and test and Each of 
       these folder contains folders for each classes.
    """
    # removing datasets folder if already exists
    if (os.path.exists(DATASET_FOLDER_PATH)):
        shutil.rmtree(DATASET_FOLDER_PATH)
    # prepairing datasets folders
    for cat in ["train", "val", "test"]:
        for c in CLASSES:
            os.makedirs(f"{DATASET_FOLDER_PATH}/{cat}/{c}", exist_ok=True)
            os.makedirs(f"{DATASET_FOLDER_PATH}/{cat}/{c}", exist_ok=True)


def move_file(file: dict):
    """This function moves file from source to destination path if file exists.

    Args:
        file (dict): A dictonary with source and destination path of file
    """
    src_path = file["src_path"]
    dest_path = file["dest_path"]
    if (os.path.exists(src_path)):
        shutil.move(src_path, dest_path)


def move_files(files: list):
    """This function moves files from source to destination path

    Args:
        files (list): _description_
        type (str): _description_
    """
    with Pool(CPU_COUNT) as p:
        p.map(move_file, tqdm(files, "Moving Files..."))


def map_dir(folder: str):
    """This function maps the given folder.

    Args:
        folder (str): Folder Name to which all the files will be mapped 
    """
    with open(f'{TEMP_FOLDER_PATH}/{folder}.txt') as file:
        files = []
        for i, line in enumerate(file):
            s_line = line.split()
            file_name = f'cxr_{i}.png'
            files.append({
                "src_path": os.path.join(TEMP_FOLDER_PATH, folder, s_line[1]),
                "dest_path": os.path.join(DATASET_FOLDER_PATH, folder, s_line[2], file_name),
                "class": s_line[2]
            })
        move_files(files=files)


def prepare_dataset():
    """This function prepares the dataset folder
    """
    prepare_dataset_folders()

    for folder in tqdm(["train", "test"], "Processing Dirs..."):
        map_dir(folder)
