# Reviewer 6
We appreciate the reviewers comments and recommendations. We especially appreciate the reviewers reference to the fact that existing works are “oftentimes coarse” and highlights that the presented behaviors are “very fine-grained and therefore challenging to classify.” We hope to address some of the reviewers concerns and suggestions below.

> Overall, the behavior classification seems to perform relatively badly (e.g. mAP is mostly below 0.2). It makes sense to me that this is a challenging dataset, though I am not convinced that the baseline results given in the paper are (close to) optimal. Did you try fine-tuning (potentially lightweight CNN) behavior classifiers?

Thank you for the suggestion! We chose to focus on the frozen backbone + MLP classifier setting to align with the capabilities of vision foundation models following work like VideoGLUE (Yuan et al.) and VideoPrism (Zhao et al.). These models generate general-purpose features that are cheaply adaptable to different tasks which makes an interesting lightweight baseline for behavior classification. 

We agree with your assessment that we can improve our evaluation method and results, based on your feedback, we performed the following additional experiments:
1. We trained a lightweight classification head on a ResNet50 backbone. 
2. We fully fine-tuned ResNet50 and DINOv2.
3. We ran several new experiments on all backbones using different loss functions (to address class imbalance). 

We present a subset of results below for easy comparison, and will include the full set of training configurations in the revised paper and upon request. 

### To-Do: If it makes more sense I'm debating changing the common configuration to be: balanced, uniform. I think this probably makes more sense because we would be directly comparing with the original paper. We can then have a separate table for the different loss metrics. 
### CoralCam
| Backbone  | Sampler  | Tuning Method | Weight Config                         | F1 Macro |   mAP   |
|-----------|----------|----------|----------------------------------------|----------|---------|
| DINOv2 ViT-L | balanced | Frozen + MLP   | Focal Loss (α=0.5, γ=1)               | 0.1224   | 0.0933  |
| DINOv2 ViT-L | balanced | Full   | Focal Loss (α=0.5, γ=1)               | To-Do   | To-DO  |
| VideoMAE     | balanced   | Frozen + MLP    | Focal Loss (α=0.75, γ=5)            | 0.0777   | 0.0703  |
| ResNet50     | balanced   | Frozen + MLP    | Focal Loss (α=0.5,γ=1)            | 0.1043   | 0.0874  |
| ResNet50     | balanced   | Full    | Focal Loss (α=0.5,γ=1)            | **0.2089**  | **0.1906**  |
#### Note: Ecologists initially defined 5 behavior classes, but one never appeared in the videos. Our original evaluation included this unused "null" class, reducing average scores by ~20%. The updated evaluation uses only the 4 classes described in the paper.

### FishFollow
| Backbone  | Sampler  | Tuning Method | Weight Config                         | F1 Macro |   mAP   |
|-----------|----------|----------|----------------------------------------|----------|---------|
| DINOv2 ViT-L | balanced | Frozen + MLP     | Focal Loss (α=0.5, γ=1)             | 0.1785  | **0.1976**  |
| DINOv2 ViT-L | balanced | Full     | Focal Loss (α=0.5, γ=1)             | To-Do  | To-Do  |
| VideoMAE     | balanced   | Frozen + MLP     | Focal Loss (α=0.75, γ=5)          | **0.1904**   | 0.1824  |
| ResNet50     | balanced   | Frozen + MLP     | Focal Loss (α=0.5,γ=1)            | 0.155   | 0.1742  |
| ResNet50     | balanced   | Full     | Focal Loss (α=0.5,γ=1)            | 0.1765   | 0.186  |

Based on the results, training a lightweight classification head on top of a ResNet50 backbone yields performance that is slightly worse or comparable to DINOv2 and VideoMAE across both datasets. However, fully fine-tuning the ResNet50 backbone significantly improves performance—particularly on CoralCam, where mAP increases from ~0.1 to ~0.2. Smaller but consistent gains are also observed on FishFollow. These results suggest that the MLP classification head in our original baseline was a limiting factor.

Thanks to the reviewer's suggestions, we will add these additional results to the paper to show the current gap in ability between frozen and fully fine-tuned models. While we could not fully finetune VideoMAE during the rebuttal period due to computational constraints, we will be sure to include those results in the camera-ready version. 

> The evaluation is relatively limited. If the dataset is challenging, probing what exactly makes it challenging would be interesting (e.g. using confusion matrices to see whether some behaviors are frequently confused). Relatedly, with respect to Table 1, what makes this and other datasets “Complex”? There is also no evaluation of tracking performance. This is obviously not the main point of the paper but including some MOT metrics would be informative.

## To-Do
- Get confusion matrices for relevant class groups (Jerome)
- Frequency vs. Performance Table (Ethan tomorrow)
- What makes this and other datasets complex? (Abby)
- Tracking performance (Abby)

### Want to include something like this: 
Also based on per-class performance (add some numbers here), some behaviors are very sparse naturally, and focusing on more efficient adaptation techniques could help improve the performance on our benchmark in the future. We want to note that sparse behaviors is a natural cause of studying fish in the wild, and these behaviors are expert labelled by domain experts as part of their ecological data analysis, so reflects real-world conditions. (Even if general-purpose approaches don't work, it's important for us to work towards finding a solution), Our work highlights this gap.


> How informative are sliding windows of 16 frames at 60FPS video, especially for T.I. = 1? My intuition would be that many behaviors need more context to classify complex behaviors. Is there a reason to not use larger windows / higher T.I.s?

A sliding window of 16 frames at 60 fps is indeed informative and sufficient for this dataset. The majority of behaviors in this context occur extremely quickly (e.g biting and aggression takes XXms) which is why we use a small window and temporal interval. 

- Try to get more diverse T.I Results. to-do: put the ti8 results in. 

> Stereo videos were collected but seem to have played no significant role in the evaluation. It makes sense to me that stereo information might not be particularly useful for behavior classification, but motivating why stereo video was collected in the first place or highlighting the potential of stereo video for future work would go a long way.

We thank the reviewer for this point. Currently all our benchmarks and behavior analysis pipeline only use 1 camera view, and we will update the paper to clarify this. We agree that the current behavior classifier benchmarks only require monocular videos. Stereo was initially recorded for the CoralCam dataset since it requires a lot of resources to go on location and set up the equipment, so any redundant camera views are helpful. This does have the potential in the future for work on 3D, and the stereo pair data for each video will be included in the final version of the dataset. Furthermore, we will more clearly describe the motivation for stereo pair collection more clearly in the revised paper.

>Lines 34-36: did it take 40h / 320h to annotate the entire datasets? Potentially including annotation effort per hour of video would be more informative



> The code does unfortunately not contain a minimal, self-sufficient example (e.g. in a JupyterNotebook) of how to parse the data and map the annotations to bounding boxes to human-readable annotations. Grammar, formatting and upper/lower case consistency should be improved. Use common backbone names (e.g. DINOv2 ViT-B / ViT-L) rather than “dino_large” etc.

We thank the reviewer for the the following suggestions and will be sure to include them in the final version of the paper. 