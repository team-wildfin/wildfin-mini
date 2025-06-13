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

## ⚙️ Dataset Configuration

All dataset settings are managed through `config/actual/dataset.yml`.

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

## ✅ Step 3: Match Frame Counts with Annotations

Ensure that frame counts and TSV annotation lines are aligned.

```bash
python data/validation/match_labels.py
```

---

## 🧠 Step 4: Extract Features with Pretrained Models

Extract features using models like DINOv2, VideoMAE, etc.

```bash
python scripts/extract_features.py
```

---

## 🧱 Step 5: Apply Sliding Window and Generate Labels

Use sliding window settings to generate `(clip, label)` pairs.

```bash
python scripts/slide_window.py
```

---

## 🔍 Step 6: Validate Feature and Label Integrity

Checks for mismatches and logs stats; important when using SLURM.

```bash
python data/validation/validate.py
```

---

## 🏋️ Step 7: Train Classification Models

Train a linear/probe classifier on top of extracted features.

```bash
python scripts/train.py
```

---

## 🧪 Step 8: Evaluate on Test Set

Run inference on the test set using wandb artifacts from training.

```bash
python scripts/evaluate.py
```

---

## 📊 Step 9: Export Evaluation Results

Aggregate results into structured tables for analysis and reporting.

```bash
python scripts/export.py
```

---

## 🧼 Step 10: Clean Tables for Paper Use

Produce paper-ready tables and clean summaries.

```bash
python scripts/clean.py
```

---

## 📌 Notes

- Dataset splits are located in:  
  `data/organization/splits/<dataset>.json`

- All global settings (e.g., dataset name, model, paths, sliding style) should be configured in:  
  `config/actual/dataset.yml`

- Sliding window styles are defined in:  
  `config/sliding_styles.yml`

- To run jobs in parallel using SLURM, set `PARALLEL = True` inside applicable scripts.


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
- Load the test dataset from `config/actual/dataset.yml`
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

The script skips incompatible sliding styles using the whitelist in `config/actual/dataset.yml`.

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
