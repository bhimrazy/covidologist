import typer
# from src.build_dataset import prepare_dataset
# from src.pytorch.train import train_and_save_model
from src.pytorch.utils import check


def main():
    # prepare_dataset()
    # datasets
    # print(image_datasets)
    # train model
    # train_and_save_model()
    check()


if __name__ == "__main__":
    typer.run(main)
