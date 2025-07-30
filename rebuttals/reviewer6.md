# Reviewer 6
We appreciate the reviewers comments and recommendations. We especially appreciate the reviewers reference to the fact that existing works are “oftentimes coarse” and highlights that the presented behaviors are “very fine-grained and therefore challenging to classify.” We hope to address some of the reviewers concerns and suggestions below.

> Overall, the behavior classification seems to perform relatively badly (e.g. mAP is mostly below 0.2). It makes sense to me that this is a challenging dataset, though I am not convinced that the baseline results given in the paper are (close to) optimal. Did you try fine-tuning (potentially lightweight CNN) behavior classifiers?

Thank you for the suggestion! We chose to focus on the frozen backbone + MLP classifier setting to align with the capabilities of vision foundation models following work like VideoGLUE (Yuan et al.) and VideoPrism (Zhao et al.). These models generate general-purpose features that are cheaply adaptable to different tasks which makes this approach interesting lightweight baseline for behavior classification. 

We agree with your assessment that we can improve our evaluation method and results, based on your feedback, we performed the following additional experiments:
1. We trained a lightweight classification head on a ResNet50 backbone. 
2. We fully fine-tuned ResNet50 and DINOv2.
3. We ran several new experiments on all backbones using different loss functions, focal loss and inverse frequency weighted BCE. 

We present a subset of results below for easy comparison, but we ran experiments on all configurations of models, sampling, tuning, loss, etc. and will include the full set of training configurations in the revised paper and upon request. 

### CoralCam
| Backbone  | Sampler  | Tuning Method | Loss Config                         | F1 Macro |   mAP   |
|-----------|----------|----------|----------------------------------------|----------|---------|
| DINOv2 ViT-L | balanced | Frozen + MLP   | Uniform BCE              | 0.1152   | 0.08887  |
| DINOv2 ViT-L | balanced | Full   | Uniform BCE              | 0.1728   | 0.1819  |
| VideoMAE     | balanced   | Frozen + MLP    | Uniform            | 0.06569   | 0.05743  |
| ResNet50     | balanced   | Frozen + MLP    | Uniform            | 0.0970   | 0.0779  |
| ResNet50     | balanced   | Full    | Uniform           | 0.1873  | 0.1890  |
| ResNet50     | random   | Full    | Focal Loss (α=0.75, γ=5.0)             | **0.2214**  | **0.21925**  |
#### Note: Ecologists initially defined 5 behavior classes, but one never appeared in the videos. Our original evaluation included this unused "null" class, reducing average scores by 20%. The updated evaluation uses only the 4 classes described in the paper.

### FishFollow
| Backbone  | Sampler  | Tuning Method | Loss Config                         | F1 Macro |   mAP   |
|-----------|----------|----------|----------------------------------------|----------|---------|
| DINOv2 ViT-L | balanced | Frozen + MLP     | Uniform             | 0.1610 | 0.1890   |
| DINOv2 ViT-L | balanced | Full     | Uniform             | 0.1751  | 0.1870  |
| VideoMAE     | balanced   | Frozen + MLP     | Uniform          | 0.1680   | 0.1920  |
| ResNet50     | balanced   | Frozen + MLP     | Uniform            | 0.1550   | 0.1742  |
| ResNet50     | balanced   | Full     | Uniform            | 0.1765  | 0.1860  |
| ResNet50     | random   | Full     | Focal Loss (α=0.75, γ=5.0)            | **0.2044**   | **0.2013** |

Based on the results, training a lightweight classification head on top of a ResNet50 backbone yields performance that is slightly worse or comparable to DINOv2 and VideoMAE across both datasets. However, fully finetuning the ResNet50 backbone (and also DINOv2 ViT-L) significantly improves performance—particularly on CoralCam, where performance approximately doubles. Smaller but consistent gains are also observed on FishFollow when comparing full finetuning and frozen backbone + MLP finetuning. These results suggest that the MLP classification head in our original baseline was a limiting factor. We also can see further improvements in performance by applying Focal Loss with random samapling. 

Thanks to the reviewer's suggestions, we will add these additional results to the paper to show the current gap in ability between frozen and fully fine-tuned models. While we could not fully finetune VideoMAE during the rebuttal period due to computational constraints, we will be sure to include those results in the camera-ready version. 

> The evaluation is relatively limited. If the dataset is challenging, probing what exactly makes it challenging would be interesting (e.g. using confusion matrices to see whether some behaviors are frequently confused). Relatedly, with respect to Table 1, what makes this and other datasets “Complex”? There is also no evaluation of tracking performance. This is obviously not the main point of the paper but including some MOT metrics would be informative.

We thank the reviewer for their suggestions and agree that some visual analysis would make interpreting the results easier. Since our data has multiple labels, we can only show confusion matrices per-class and will include all of these results in the supplementary material. We show per-class confusion matrices on CoralCam for the best performing ResNet50 model with full finetuning vs. frozen backbone + MLP finetuning. The ***first*** number in each cell is for full finetuning and the second number is for MLP. 

|  Class: Mouth Not-Visible | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | ***5347***, 3221             | ***2115***, 4241            |
| Actual Negative | ***5389***, 6026               | ***60261***, 59624           |

| Class: Feeding  | Predicted Positive | Predicted Negative |
|-----------------|------------------|---------------------|
| Actual Positive | ***136***, 2     | ***503***, 637      |
| Actual Negative | ***138***, 1     | ***72335***, 72472  |

| Class: Being Charged | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | ***0***, 0            | ***42***, 42  |
| Actual Negative | ***13***, 0              | ***73057***, 73070|

| Class: C-Turn | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | ***0***, 0           | ***8***, 8    |
| Actual Negative | ***0***, 0          | ***73104***, 73104  |

These results show that full fine-tuning improves performance on more frequent behaviors (e.g., Mouth Not-Visible, Feeding), while both approaches (and all other models and configurations) consistently struggle with the very sparse but ecologically important behaviors (Being Charged, C-Turn). This sparsity reflects real-world conditions and poses a significant challenge for current models. While our work does not solve this issue, we hope it motivates future research into more effective methods for handling rare, fine-grained behaviors in the wild.

> How informative are sliding windows of 16 frames at 60FPS video, especially for T.I. = 1? My intuition would be that many behaviors need more context to classify complex behaviors. Is there a reason to not use larger windows / higher T.I.s?

A sliding window of 16 frames at 60 fps is indeed informative and sufficient for this dataset. The majority of behaviors in this context occur extremely quickly (e.g biting and aggression takes XXms) which is why we use a small window and temporal interval. While DINOv2 and ResNet50 methods use per-frame analysis, we perform experiments using T.I = 8 on CoralCam for VideoMAE w/ balanced sampling and Uniform BCE loss shown below. The results generally support the original hypothesis as the mAP is lower.

| Backbone| T.I | F1 Macro |   mAP   |
|---------|-----------|-----|----------|
|VideoMAE | 1   | 0.06569  | 0.05743 |
|VideoMAE | 8   | 0.06716  | 0.04925 |


> Stereo videos were collected but seem to have played no significant role in the evaluation. It makes sense to me that stereo information might not be particularly useful for behavior classification, but motivating why stereo video was collected in the first place or highlighting the potential of stereo video for future work would go a long way.

We thank the reviewer for this point. Currently all our benchmarks and behavior analysis pipeline only use 1 camera view, and we will update the paper to clarify this. We agree that the current behavior classifier benchmarks only require monocular videos. We recorded stereo for the CoralCam dataset because it requires a lot of resources to go underwater on location and set up the equipment, so any redundant camera views are helpful. The stereo data does have potential for work related to 3D reconstruction of underwater environments in the future, and the stereo pair data for each video will be included in the final version of the dataset. Furthermore, we will more clearly describe the motivation for stereo pair collection more clearly in the revised paper.

>Lines 34-36: did it take 40h / 320h to annotate the entire datasets? Potentially including annotation effort per hour of video would be more informative



> The code does unfortunately not contain a minimal, self-sufficient example (e.g. in a JupyterNotebook) of how to parse the data and map the annotations to bounding boxes to human-readable annotations. Grammar, formatting and upper/lower case consistency should be improved. Use common backbone names (e.g. DINOv2 ViT-B / ViT-L) rather than “dino_large” etc.

We thank the reviewer for the the following suggestions and will be sure to include them in the final version of the paper. 