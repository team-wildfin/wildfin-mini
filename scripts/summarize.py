from config.data.datasets import CORALCAM, FISHFOLLOW
from vision_bench.typing.types import  LocalDataset, Split
from vision_bench.management.manager import SourceManager
import os 
import pandas as pd
import numpy as np
import json

def summarize(dataset: LocalDataset, destination: str = "summary.json"): 
    source_manager = SourceManager(dataset.path)
    print(f"Dataset: {dataset.name}")
    cumulative = np.zeros(len(dataset.categories))
    for split in dataset.splits:
        subsets = source_manager.list_subsets(split.name)
        print(f"  Split: {split.name}, Subsets: {subsets}")
        for subset in subsets:
            path = source_manager.subset_path(split.name, subset)
            #read in the tsv file 
            label_file = os.path.join(path, f"{subset}.tsv")

            if not os.path.exists(label_file):
                label_file = os.path.join(path, f"{subset}.txt")

            if not os.path.exists(label_file):
                print(f"Label file not found for {split.name} {subset}")
                continue

            # sum up the columns of the tsv file and print out the total for each category
            # there's no header
            df = pd.read_csv(label_file, sep="\t")
            col_sums = df.sum().to_numpy()
            cumulative += col_sums

    result = {category: int(count) for category, count in zip(dataset.categories, cumulative)}
    #save result 
    with open(destination, "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    summarize(CORALCAM, destination="coralcam_summary.json")
    summarize(FISHFOLLOW, destination="fishfollow_summary.json")

