import os
import urllib.request
import zipfile
import tarfile


def download_hci_dataset(output_dir="./data/hci"):
    print("Downloading HCI Dataset...")
    print("The HCI dataset can be downloaded from: http://lightfield-analysis.net/dataset/")
    print("Please download the dataset manually and extract it to:", output_dir)
    print("\nThe dataset contains synthetic light field images with ground truth depth maps.")
    print("Scenes include: Dino, Sideboard, Backgammon, Pyramids, etc.")


def download_stanford_lytro(output_dir="./data/lytro"):
    print("Downloading Stanford Lytro Archive...")
    print("The Stanford Lytro Archive can be downloaded from: http://lightfields.stanford.edu/")
    print("Please download the dataset manually and extract it to:", output_dir)


def prepare_data_structure():
    data_dir = "./data"

    subdirs = [
        "hci/train",
        "hci/test",
        "hci/val",
        "slf/train",
        "slf/test",
        "lytro/train",
        "lytro/test"
    ]

    for subdir in subdirs:
        path = os.path.join(data_dir, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")


def main():
    print("=" * 60)
    print("Data Preparation for DispNet-OccNet")
    print("=" * 60)

    print("\n1. Creating data directory structure...")
    prepare_data_structure()

    print("\n2. Dataset Download Information:")
    print("-" * 60)
    download_hci_dataset()
    print()
    download_stanford_lytro()

    print("\n3. Data Directory Structure:")
    print("-" * 60)
    print("""
After downloading and extracting the datasets, your directory structure should look like:

./data/
├── hci/
│   ├── train/
│   │   ├── dino/
│   │   │   ├── 00/
│   │   │   │   ├── 00.png
│   │   │   │   ├── 01.png
│   │   │   │   └── ...
│   │   │   ├── 01/
│   │   │   └── ...
│   │   ├── sideboard/
│   │   └── ...
│   └── test/
│       └── ...
├── slf/
│   └── (sparse light field data)
└── lytro/
    └── (Stanford Lytro data)
    """)

    print("\n4. Quick Start:")
    print("-" * 60)
    print("After preparing the data, you can start training:")
    print("  python train.py --data_dir ./data/hci/train --batch_size 4")
    print("\nOr with a custom config:")
    print("  python train.py --config configs/dense_lf.yaml --data_dir ./data/hci/train")


if __name__ == "__main__":
    main()
