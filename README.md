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