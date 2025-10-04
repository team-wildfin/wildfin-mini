import os
import numpy as np
import json
import yaml
from config.datasets import DATASETS

def is_txt_or_tsv(filename):
    return filename.endswith(".txt") or filename.endswith(".tsv")

def load_file(path):
    try:
        return np.loadtxt(path, delimiter=None)
    except ValueError:
        return np.genfromtxt(path, delimiter=None)

def walk_and_collect_arrays(root_dir):
    arrays = []
    expected_num_columns = None
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if is_txt_or_tsv(fname):
                full_path = os.path.join(dirpath, fname)
                print(f"Loading: {full_path}")
                data = load_file(full_path)
                if data.ndim == 1:
                    data = data.reshape(1, -1)

                if expected_num_columns is None:
                    expected_num_columns = data.shape[1]
                else:
                    assert data.shape[1] == expected_num_columns, (
                        f"Inconsistent column count in {full_path}: "
                        f"expected {expected_num_columns}, got {data.shape[1]}"
                    )

                arrays.append(data)
    return arrays

def count_column_positives(array):
    return np.sum(array > 0, axis=0).tolist()

def get_pos_freq(root_dir):
    arrays = walk_and_collect_arrays(root_dir)
    if not arrays:
        print("No .txt or .tsv files found.")
        return
    combined = np.vstack(arrays)
    positive_counts = count_column_positives(combined)
    return positive_counts

def main(root_dir, output_json):
    arrays = walk_and_collect_arrays(root_dir)
    if not arrays:
        print("No .txt or .tsv files found.")
        return

    combined = np.vstack(arrays)
    positive_counts = count_column_positives(combined)

    with open(output_json, "w") as f:
        json.dump(positive_counts, f, indent=2)

    print(f"Saved positive column counts to {output_json}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Count positive values per column in .txt/.tsv files.")
    parser.add_argument("--dataset", help="Root directory to search for text files", required=True)
    parser.add_argument("--output", default="positive_counts.json", help="Output JSON filename")
    args = parser.parse_args()
    dataset = DATASETS[args.dataset]
    root = dataset.path + '/train' # this is used for weighted loss during training, thus we only need the train split
    main(root, args.output)
