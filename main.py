import typer
from src.build_dataset import prepare_dataset


def main():
    prepare_dataset()


if __name__ == "__main__":
    typer.run(main)
