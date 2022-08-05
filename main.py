import typer
from src.build_dataset import prepare_dataset
from src.pytorch.dataloader import image_datasets


def main():
    # prepare_dataset()
    # datasets
    print(image_datasets)


if __name__ == "__main__":
    typer.run(main)
