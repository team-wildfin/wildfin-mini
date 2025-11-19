# 🐟 VisionBench

**VisionBench** is a modular and scalable video classification framework for multiclass tasks. Designed for dynamic, efficient experimentation, it offers DMVR-style video ingestion, flexible model composition, and support for large-scale training over any video dataset. Built in PyTorch, it's fully configurable and integrates seamlessly with Weights & Biases for tracking and artifact management.

---

## 🔧 Key Features

- **Dynamic Video-to-Clip Conversion**  
  Read raw `.mp4` videos and produce sliding-window clips with spatial/temporal patching.

- **Modular Model Architecture**  
  Combine foundation backbones (e.g., DINO, CLIP), pooling layers, and classifiers using broadcastable components.

- **Scalable Pipeline**  
  Distributed processing over shards, label-only computation for fast statistics, and precomputed feature loading.

- **WANDB Integration**  
  Training logs, evaluation metrics, and model checkpoints tracked as artifacts.

---

## 📦 Installation

Unzip the zip file, then
```bash
cd wildfin-mini
conda env create -f environment.yml
```

## ⚙️ Dataset Configuration

All dataset settings are managed through `config/data/datasets.py`.
Each dataset block defines:

- `doi`: Link to the official dataset source (Dataverse)
- `path`: Location of the organized video + annotation directory
- `precomputed_path`: Where extracted features and sliding label outputs will be stored
- `label_type`: Currently supports `onehot` for multi-class classification
- `categories`: Full list of behavior/action labels  
  > 🧠 These are **domain-specific categories** defined by marine biologists and used throughout training, evaluation, and reporting
- `splits`: Defines which sliding styles to use for `train`, `val`, and `test`

Example (excerpt for `fishfollow`):

```yaml
fishfollow:
  doi: https://doi.org/10.7910/DVN/QN66Z8
  path: '/path/to/organized'
  precomputed_path: '/path/to/precomputed'
  label_type: onehot
  categories:
    - Other behavior
    - Medium bites
    - High bites
    ...
  splits:
    train:
      sliding_styles:
        - frames
        - sliding_window
    val:
      sliding_styles:
        - frames
    test:
      sliding_styles:
        - test_frames
```

# 🐟 Reproducing Experiments in **WildFin**

This guide outlines how to reproduce all experiments reported in the WildFin benchmark.
Follow the 3 steps below to download and organize the data in standard format, then follow ```guide.ipynb``` to run the experiments. 
One can use Wildfin platform to run video benchmarking experiments on custom benchmarks after organizing files in the same format! 

---

## 🔽 Step 1: Download Raw Videos and Annotations

Use the following script to download raw data from Dataverse.

```bash
python data/action/download/download.py
```

---

## 📁 Step 2: Organize Dataset Structure

Run dataset-specific organization scripts to match the required directory layout.

```bash
python data/organization/scripts/coralcam.py
python data/organization/scripts/fishfollow.py
```

---

## ✅ Step 3: Sanity Check Frame Counts matches Annotations

Ensure that frame counts and TSV annotation lines are aligned.

```bash
python data/validation/match_labels.py
```

---