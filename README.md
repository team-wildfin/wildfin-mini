# 🐟 WildFin

**WildFin** is a modular and scalable video classification framework for multiclass tasks. Designed for dynamic, efficient experimentation, it offers DMVR-style video ingestion, flexible model composition, and support for large-scale training over any video dataset. Built in PyTorch, it's fully configurable and integrates seamlessly with Weights & Biases for tracking and artifact management.

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

# Reproducing Experiments in **WildFin**

**Note:** The datasets have internal nicknames in the codebase:  
- **FishFollow** is referred to as `"mike"`  
- **CoralCam** is referred to as `"abby"`

---

## 🛠️ Design Overview

The `scripts/` folder orchestrates the end-to-end machine learning pipeline by executing logic defined in:
- `training/`
- `evaluation/`
- `data/action_scripts/`

⚠️ **Important:** Before executing any script, **review and configure the global variables** defined at the top of the file.  
These variables determine critical settings such as dataset name, model architecture, data paths, and filtering parameters.  
**Do not execute scripts without verifying these settings** — they must align with your experimental goals.

To enable SLURM-based distributed processing, set the `PARALLEL` variable to `True`. This will launch one job per video via the cluster scheduler. For local testing and debugging, set `PARALLEL` to `False`. A common workflow is to test locally first, then scale up via SLURM for full-batch processing.

---

## 📌 Steps

### 1. Download the Data

Please find the datasets from these links:

CoralCam:
```
https://dataverse.harvard.edu/previewurl.xhtml?token=b35e20a9-8204-4538-92bc-46f0fe310d61
```

Fishfollow:
```
https://dataverse.harvard.edu/previewurl.xhtml?token=ec755cda-1309-4de6-be2a-0ab2555b5e5f
```
RiverCam:
```
https://dataverse.harvard.edu/previewurl.xhtml?token=e465cfc9-708a-4f5b-b771-72fa3d4bd94d
```

Once you download the datasets, organize the dataset directory as follows:

```
<root>/
  <split>/            # train / val / test
    <video_id>/   
      <video_id>.mp4  # video 
      <video_id>.tsv  # per-frame annotation
```

---

### 2. Set the Root Directory

Edit `config/dataset.yml` and set the `path` field for `"mike"` and `"abby"` to the root directory you created above.

---

### 3. Configure Sliding Styles
- In the same YAML file, set the desired `sliding_style` for each dataset.
- Definitions for each style are found in `config/sliding_styles.yml`.

> ✅ The default configurations reproduce the settings used in the paper.

---

### 4. Generate (Clip, Label) Pairs
Run:

```bash
python scripts/preprocess_sliding_window.py
```

This applies the sliding window to the videos and outputs:
- `(clip, label)` pairs, or
- just labels (if you prefer to extract features later without storing raw clips).

---

### 5. Extract Model Features
Run:

```bash
python scripts/extract_features.py
```

This extracts model features for each clip.  
- Input: Clips from the previous step or loaded via `DatasetBuilder`.
- Output:
  - Features stored in `<model>_features/`
  - File naming:
    - Features: `<video_id>_<frame_id>.npy`
    - Labels: `<video_id>.txt`

---

### 6. Train Classification Heads
Run:

```bash
python scripts/train.py
```

This script trains classification heads on precomputed features.

> 🔒 Ensure that the same model used for feature extraction is set in the global config.  
> 🧭 Sign in to your Weights & Biases (wandb) account to monitor training and manage artifacts.

---

### 7. Evaluate on the Test Set
Run:

```bash
python scripts/evaluate.py
```

This script evaluates each training configuration on its corresponding test set.  
- Training–test mapping is defined in `config/dataset.yml`.
- Evaluation runs are logged to dedicated `wandb` projects and reference their original training runs.

---

### 8. Export Results as Tables
Run:

```bash
python scripts/export.py
```

This retrieves evaluated runs based on filters defined in the script and produces CSV summaries including:
- Aggregated metrics
- Subgroup scores
- Per-class metrics

---

### 9. Clean Final Results for the Paper
Run:

```bash
python scripts/clean.py
```

This script filters and formats relevant columns into paper-ready result tables.

---

# Software Architecture Overview:  

## 🗂 Dataset Formats
Choose the corresponding source for your data and update the get_source in fish_benchmark.data.dataset.py and mount the correct source type on the folder.  

### 1. Frame-Annotated
Each `.mp4` file is paired with a `.tsv` file containing frame-level labels.

```
data/
├── video_001.mp4
├── video_001.tsv
└── ...
```

### 2. Video-Annotated
Each `.avi` file is paired with a `.txt` file containing a single class label.

```
data/
├── classA/
│   ├── vid123.avi
│   └── vid123.txt
```

### 3. Precomputed Features
Used for fast training on saved numpy features.

```
data/
├── frames/
│   ├── video_001_000001.npy
│   └── ...
├── labels/
│   └── video_001.tsv
```

---

## 🔁 Sliding Window Sampling

Sampling is controlled via YAML config (`sliding_style.yml`) with the following parameters:

| Parameter              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `window_size`          | Total number of frames in a window                                          |
| `samples_per_window`   | How many frames are sampled per window (must divide `window_size`)          |
| `tolerance_region`     | Midpoint ± region for label aggregation                                     |
| `step_size`            | Sliding step between windows                                                |
| `data_ndim`            | Dimensionality of data (`3` for images, `4` for video)                      |
| `patch_type`           | Either `absolute` or `relative`                                             |
| `patch_h`, `patch_w`   | Patch layout parameters                                                     |
| `temporal_sample_interval` | Downsampling rate in time                                             |
| `MAX_BUFFER_SIZE`      | Shuffle buffer size for streaming                                           |
| `shuffle`              | Whether to return randomly ordered outputs                                  |

---

## 🧬 DatasetBuilder API

```python
from fish_benchmark.data.dataset import DatasetBuilder

dataset = DatasetBuilder(
  path=SOURCE,
  dataset_name=DATASET,
  style=SLIDING_STYLE, 
  transform=input_transform,
  precomputed=PRECOMPUTED
).build()
```

Options:
- `only_labels=True` for fast label stats
- `min_ctime` to filter stale files
- `transform` to inject Torch preprocessing

---

## 🧠 ModelBuilder API

```python
from fish_benchmark.models import ModelBuilder

hidden_size = ModelBuilder().set_backbone(MODEL).get_hidden_size()
classifier = (ModelBuilder()
    .set_hidden_size(hidden_size)
    .set_pooling(POOLING)
    .set_classifier(CLASSIFIER, 
      input_dim=hidden_size, 
      output_dim=len(train_dataset.categories))
    .set_aggregator(AGGREGATOR)
    .build()
)
```

WildFin automatically wraps each component in a `BroadcastableModule` for flexible input shape handling.

---

## 🏋️‍♀️ Training

To train a model:

```bash
python training/head.py \
    --classifier mlp \
    --pooling mean \
    --dataset abby \
    --sliding_style frames \
    --model dino \
    --sampler balanced \
    --epochs 40 \
    --lr 5e-5 \
    --batch_size 32
```

This will:
- Load the precomputed dataset (`.npy` + `.tsv`)
- Construct the model from `config/models.yml`
- Wrap everything into a `LitBinaryClassifierModule`
- Log metrics and artifacts via Weights & Biases
- Save best and latest checkpoints into `checkpoints/<run_id>/`

**Sampler options:**
- `balanced`: Multilabel-balanced sampling per class
- `random`: Uniform sampling with or without replacement

---

## ✅ Evaluation

After training, you can evaluate any saved model checkpoint using:

```bash
python evaluation/eval.py \
    --entity fish-benchmark \
    --project abby \
    --run <wandb_run_id>
```

This will:
- Load the trained model from its W&B artifact
- Load the test dataset from `config/dataset.yml`
- Run inference and log metrics
- Save predictions (`probs`, `targets`) into a JSON file
- Upload this as a W&B artifact of type `metrics`

---

## 📊 WANDB Logging

- Logs `val_loss`, `val_mAP`, `f1_macro`, `f1_micro`
- Automatically tags runs by dataset, model, pooling, classifier, etc.
- Saves best and latest checkpoints
- Logs test predictions as artifacts

---

## 📡 Distributed Computing

WildFin supports distributed and automated job execution on SLURM-based high-performance computing (HPC) clusters, making it easy to run large-scale preprocessing and training jobs in parallel over multiple shards or configuration combinations.

### 🧪 Feature Extraction

The script `data/action_scripts/precompute_sliding_window.py` extracts sliding-window inputs and their corresponding labels. This is automatically batched and submitted using:

```bash
python scripts/preprocess_sliding_window.py
```

Each job processes one video subset (e.g. a directory of frame-annotated video and label pairs) and stores:
- Inputs: `.npy` feature arrays under `precomputed/{dataset}/{sliding_style}/{split}/{subset}/inputs/`
- Labels: `.tsv` files under `precomputed/{dataset}/{sliding_style}/{split}/{subset}/labels/`

One can also only store the extracted features using scripts/extract_features.py
Which stores a the model features in 
- Inputs: `.npy` feature arrays under `precomputed/{dataset}/{sliding_style}/{split}/{subset}/{model}_features/`
- Labels: `.tsv` files under `precomputed/{dataset}/{sliding_style}/{split}/{subset}/labels/`
Both methods share the same label directory, since features extracted by different models correspond to the same labels, as long as they follow the same sliding window logic. 

#### ✅ SLURM Parallelism

The wrapper script constructs SLURM jobs via:

```bash
get_slurm_submission_command(name, output_dir, wrap_cmd, gpu=0)
```

and logs each submission to `logs/precompute_sliding_window/{dataset}/...`.

#### Example Config:
```python
TARGETS = ["abby", "mike"]
SLIDING_STYLES = ["test_frames", "test_sliding_window"]
SPLITS = ["train", "val", "test"]
```

The script skips incompatible sliding styles using the whitelist in `config/dataset.yml`.

---

### 🧠 Training Grid Search

To automate training jobs across combinations of:
- backbone models (e.g. `dino`, `dino_large`)
- classifier types (e.g. `mlp`, `linear`)
- pooling strategies (e.g. `mean`, `attention`)
- datasets (e.g. `abby`, `mike`)
- sliding styles (e.g. `frames`, `sliding_window`)
- samplers (`balanced`, `random`)

run:

```bash
python scripts/train.py
```

Each job runs:

```bash
python training/head.py --classifier mlp --pooling mean --dataset abby --sliding_style frames --model dino --sampler balanced
```

Jobs are submitted in SLURM with a call to:

```python
get_slurm_submission_command(name, output_dir, wrap_cmd, gpu_count=1)
```

#### ✅ Output Structure:
Each training run stores logs and checkpoints in:

```
logs/train/{dataset}/{sliding_style}/{model}/{pooling}/{classifier}/{sampler}/
```

This supports massive grid search scaling over GPU resources in an HPC environment with minimal code duplication.

---

### ⚠️ Notes

- `PARALLEL = True` will submit SLURM jobs instead of running locally.
- In both scripts, `get_slurm_submission_command` is a wrapper for `sbatch` with support for output/error logging.
- You can modify `config/models.yml` and `config/sliding_style.yml` to restrict which models or sliding strategies are compatible.



## 📁 Project Structure

```
wildfin/
├── data/             # Sources, patchers, sliding window datasets
├── models/           # Backbone, pooler, classifier definitions
├── scripts/          # Training & evaluation scripts
├── config/           # YAML config files for models & sliding styles
├── utils/            # Timers, I/O, helpers
├── training/         # Training entrypoints
├── evaluation/       # Evaluation entrypoints
```

---

## 🧠 Design Principles

- **Decoupled input/label streaming:** You can analyze label distributions without decoding video frames.
- **Dynamic video mounting:** Works with any directory layout as long as IDs are consistent.
- **Broadcastable processing:** Enables inference over spatial patches or temporal groups with minimal code changes.

---

## 📜 License

MIT License © 2025 WildFin Contributors
# wildfin-mini
