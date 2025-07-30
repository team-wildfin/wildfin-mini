# Reviewer 4

We thank the reviewer for their comments and suggestions and their acknowledgment that this dataset "fills a gap in publicly available, expert-annotated underwater video data”.

> 1. The benchmarking results show modest performance, e.g., mAP < 0.2 for many behaviors, suggesting the dataset is highly challenging. However, the paper does not deeply analyze why certain models fail, or propose targeted improvements.

We thank the reviewer for pointing this out. To address this, we test an additional backbone (ResNet50) and also test different finetuning methods to try and make targeted improvements. We find that ResNet50, DINO-v2, VideoMAE are all relatively comparable under the frozen backbone setting. However, we found that the biggest improvement comes from fine-tuning the entire model, which suggests that the MLP classifier head adapter may be the limiting factor in our original baseline evaluation. While we could not fully finetune VideoMAE in time during the rebuttal period, we will be sure to include those results in the camera-ready version. (Should we mention using attention head as a targeted improvement?)

### To-Do: If it makes more sense I'm debating changing the common configuration to be: balanced, uniform. I think this probably makes more sense because we would be directly comparing with the original paper. We can then have a separate table for the different loss metrics. 
### CoralCam
| Backbone  | Sampler  | Tuning Method | Weight Config                         | F1 Macro |   mAP   |
|-----------|----------|----------|----------------------------------------|----------|---------|
| DINOv2 ViT-L | balanced | Frozen + MLP   | Focal Loss (α=0.5, γ=1)               | 0.1224   | 0.0933  |
| DINOv2 ViT-L | balanced | Full   | Focal Loss (α=0.5, γ=1)               | **To-Do**   | **To-DO**  |
| VideoMAE     | balanced   | Frozen + MLP    | Focal Loss (α=0.75, γ=5)            | 0.0777   | 0.0703  |
| ResNet50     | balanced   | Frozen + MLP    | Focal Loss (α=0.5,γ=1)            | 0.1043   | 0.0874  |
| ResNet50     | balanced   | Full    | Focal Loss (α=0.5,γ=1)            | **0.2089**  | **0.1906**  |
#### Note: Ecologists initially defined 5 behavior classes, but one never appeared in the videos. Our original evaluation included this unused "null" class, reducing average scores by ~20%. The updated evaluation uses only the 4 classes described in the paper.

### FishFollow
| Backbone  | Sampler  | Tuning Method | Weight Config                         | F1 Macro |   mAP   |
|-----------|----------|----------|----------------------------------------|----------|---------|
| DINOv2 ViT-L | balanced | Frozen + MLP     | Focal Loss (α=0.5, γ=1)             | 0.1785  | **0.1976**  |
| DINOv2 ViT-L | balanced | Full     | Focal Loss (α=0.5, γ=1)             | **To-Do**  | **To-Do**  |
| VideoMAE     | balanced   | Frozen + MLP     | Focal Loss (α=0.75, γ=5)          | **0.1904**   | 0.1824  |
| ResNet50     | balanced   | Frozen + MLP     | Focal Loss (α=0.5,γ=1)            | 0.155   | 0.1742  |
| ResNet50     | balanced   | Full     | Focal Loss (α=0.5,γ=1)            | 0.1765   | 0.186  |

> 2. The dataset exhibits severe class imbalance, yet the proposed mitigation is not evaluated against alternatives.

We thank the reviewer for bringing up this point. Based on this feedback, we evaluate weighted loss methods as another avenue for addressing severe class imbalance. Specifically we evaluate focal loss and inverse frequency weighted BCE. Below we show results for each approach on CoralCam with ResNet50. We ran these experiments on all datasets and backbones and will include them in the final version of the paper and are happy to share them with the reviewer upon request. In general, the best configuration is **To-Do**.
## To-Do: Replace with DINOv2 Results?
### CoralCam
| Backbone  | Sampler  | Tuning Method | Weight Config                         | F1 Macro |   mAP   |
|-----------|----------|----------|----------------------------------------|----------|---------|
| ResNet50  | balanced | Frozen + MLP    | Focal Loss (α=0.5,γ=1)            | 0.1043   | 0.0874  |
| ResNet50     | balanced   | Full    | Focal Loss (α=0.5,γ=1)            | **0.2089**  | **0.1906**  |
| ResNet50  | random   | Frozen + MLP    | Inverse            | 0.1594   | 0.0874  |
| ResNet50     | random   | Full    | Inverse               | 0.0309  | 0.0859  |
| ResNet50  | random   | Frozen + MLP    | Inverse            | 0.1594   | 0.0874  |
| ResNet50     | random   | Full    | Inverse               | 0.0309  | 0.0859  |

## To-Do: Class Imbalance Results and Explanation. 

> 3. While the annotation effort is substantial, the paper does not discuss inter-annotator agreement or potential label noise, which is critical for behavioral datasets.

We thank the reviewer for raising this important concern and will make sure to include this discussion in the final version of the paper. 

- For the FishFollow subset, approximately 35% of videos were annotated by a second reliable annotater. We display consistency metrics between the annotators below. 
    - With a tolerance of 0.25 seconds (~15  frames):
        - Average F1 Score: 0.78
        - Median F1 Score: 0.83
        - Average Precision: 0.84
        - Average Recall: 0.75
    - With a tolerance of 0.5 seconds (~30  frames):
        - Average F1 Score: 0.86
        - Median F1 Score: 0.916
        - Average Precision: 0.925
        - Average Recall: 0.84 

These statistics reflect a generally high level of agreement among annotators. In total, annotations from 12 annotators were included during dataset development. While only one annotation set per video was used during training, we will include the additional annotator annotations in the final dataset release for transparency.

- For the CoralCam subset, annotations used for model training were provided by a single expert annotator. 
    - To assess consistency, a subset of videos was independently labeled by a second trained annotator, yielding an F1 score of 0.74 between the two.