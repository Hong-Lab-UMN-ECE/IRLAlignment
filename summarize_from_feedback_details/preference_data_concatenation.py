import sys
import os
import shutil
from datasets import load_from_disk, concatenate_datasets

def main(dataset1_path, dataset2_path, output_path):
    # Load the datasets
    for split in ["train","validation"]:
        dataset1 = load_from_disk(f"{dataset1_path}/{split}_dataset")
        dataset2 = load_from_disk(f"{dataset2_path}/{split}_dataset")

        # Concatenate the datasets
        concatenated_dataset = concatenate_datasets([dataset1, dataset2])
        concatenated_dataset = concatenated_dataset.shuffle()

        # Save the concatenated dataset to a new directory
        concatenated_dataset.save_to_disk(f"{output_path}/{split}_dataset")

        print(f'{split} datasets have been concatenated and saved to {output_path}/{split}_dataset')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <dataset1_path> <dataset2_path> <output_path>")
        sys.exit(1)

    dataset1_path = sys.argv[1]
    dataset2_path = sys.argv[2]
    output_path = sys.argv[3]

    main(dataset1_path, dataset2_path, output_path)