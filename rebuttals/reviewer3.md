# Reviewer 3

We thank the reviewer for their comments and suggestions, we were pleased to read that the author found the descriptions “clear” and the writing “well done” and that they recognize fish behavior to be an important topic.

> What are the key challenges of fish behavior recognition compared to existing behavior recognition tasks? The authors should provide a more in-depth analysis of these difficulties.

Thank you for the suggestion. Fish behavior detection, particularly in the wild, is uniquely challenging compared to behavior recognition in more commonly studied subjects like humans or terrestrial animals. Fish are non-standard subjects in computer vision—lacking consistent anatomical landmarks, often displaying reflective or translucent bodies, and exhibiting a wide range of morphologies and movement dynamics. These challenges are compounded by the uncontrolled optical conditions of underwater environments, including dynamic lighting, dim visibility, blurring, and altered color profiles. Additionally, the backgrounds in these settings are often complex and in motion, making it difficult to isolate individuals or detect subtle behavioral cues. Variability in the timescales of behavior further complicates detection: fish display actions that range from extended behaviors like foraging to the fastest known vertebrate response—sub-second escape maneuvers (Hein, 2018). The datasets used in our work add further complexity. For example, in CoralCam, many small, spatially dense individuals lead to frequent occlusions, while in FishFollow, both the camera and the focal individual are moving through intricate environments. Unlike terrestrial datasets, our recordings are 100% underwater and fully in situ, which introduces a unique combination of visual noise, motion, and ambiguity. While some of these challenges are noted in the current version of the paper [Lines 41-48], we plan to expand on them in future revisions to better highlight the specific difficulties of detecting behavior in underwater fish populations.

> The annotations in the dataset are overly simplistic, consisting only of classification labels. It is recommended to consider incorporating richer forms of annotation, such as textual descriptions, segmentation masks, and inspection-related labels.
 
Our annotation approach was designed to facilitate scientific analysis of wild animal behavior and ecology. As a result, we focused on repeatable ecologically consequential behaviors rather than textual descriptions of actions that may or may not have ecological meaning. A standard practice in ecology is to identify stereotyped, well-defined behaviors, and then to try to automatically detect instances of those behaviors. Our datasets focus on key behaviors that have been identified by ecologists in being important in influencing population and ecosystem dynamics, for example feeding. We provide a detailed description of each behavior in the supplementary material. CoralCam data includes bounding boxes and although we did not provide segmentation masks, our preliminary experiments suggest that segmenting individuals from within bounding boxes can easily be accomplished with modern segmentation methods. **Maybe we should just say that segmentation isn't as important for ecology? Otherwise they may ask us to do this...**

> The comparative methods selected in the experimental section are not sufficiently comprehensive, as only DINO-v2 and VideoMAE are included. It is recommended that the authors evaluate the performance of additional backbone models.

Thank you for the suggestion. We use DINOv2 and VideoMAE as backbones for our baseline results because they represent two diverse classes of (at the time) SOTA foundation models, image and video. **TO-Do: Find Relevant NeurIPs papers that don't have that many backbones**. Given the reviewer's recommendations, we run experiments on ResNet50 to establish baseline results on CNN backbones. While we are unable to perform experiments on other large backbones during the rebuttal period as we are limited by computation, we plan to establish baselines on the newly released SOTA VideoPrism. We are also happy to run baselines on any other backbones that the reviewer suggests. Additionally, we perform further experiments with other loss functions that are designed to handle the class imbalance present in the dataset. Because there are many combinations of these hyperparameters (backbone, sampling method, loss function, tuning method), we only show a subset of them in this response but we are happy to share more results upon request.

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