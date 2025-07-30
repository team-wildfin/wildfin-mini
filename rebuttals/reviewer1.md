# Reviewer 1
We thank the reviewer for their comments and suggestions:

> 1. The paper mentions that the data collection covers various environments (e.g., coral reefs, rivers), but does not specify the number of videos or duration distribution for each type of environment. It is recommended to provide statistical information on data distribution (such as the proportion of frames from each environment) to help readers better understand the dataset's balance.

We thank the reviewer for their recommendation. Table 1 reports the total duration (in hours) for each of the datasets. To make the data distribution clearer, we will additionally convert these durations to approximate frame counts using the dataset-specific frame rates and include these values in the revised table.
Details such as the number of videos, resolutions, and frame counts for each environment subset (CoralCam, FishFollow, RiverCam) are currently provided across Section 3 and in the supplementary material. (**Should we add specific line numbers?**) In response to this comment, we will consolidate this information into a single summary table to improve clarity in the final version.

> 2. To provide a more comprehensive understanding of the proposed dataset, it is suggested to include more detailed statistics in Table 1, such as the total number of frames, average video length, target categories, and annotation types (e.g., bounding boxes, masks).

We appreciate the reviewer’s suggestion. Table 1 was designed to emphasize the key differences in purpose across the related datasets. In the final version, we will expand the supplementary material to include more detailed statistics and comparisons to provide a more comprehensive overview of the problem.

> 3. The annotation process for behaviors (such as “feeding, charging, being charged, c-turn”) is a highlight, but the specific annotation standards or consistency verification methods (e.g., inter-annotator agreement metrics) are not clearly explained.

We thank the reviewer for pointing this out and we provide annotation consistency verification metrics below and will provide it in the supplementary material. 
 
- For the FishFollow subset, approximately 35% of videos were annotated by a second reliable annotater. We report consistency verification metrics below. 
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


> 4. The paper does not propose any new algorithms. 

While the paper does not propose any new methods, we evaluate different SOTA models and various approaches on this task. By running evaluation on a vision foundation model (DINOv2) and video foundation model (VideoMAE), we provide baseline evaluations across a diverse group of models. Our tasks are motivated by real ecological data analysis workflows, which are not well explored in ML benchmarks and also follow methods from related work in animal behavior analysis in videos like VideoPrism (Zhao, et al.) and VFM for Animal Behavior Analysis (Sun, et al.). 

> 4. Additionally, it is suggested that the authors explore more tasks to demonstrate the versatility of the dataset, such as fish video object tracking and segmentation.

The CoralCam subset of this dataset includes model weights and annotations for object detection. Additionally, this dataset contains individual tracks which offers further exploration of multi-object tracking capabilities for underwater datasets. We will make the ecological pipeline and all artefacts more clear in the final version of the paper to highlight the versatility of this dataset.