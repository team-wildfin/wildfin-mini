import pandas as pd
import os
DATASETS = ['mike', 'abby']
TOLS = [0, 1, 3, 5, 7]
OUTPUT_PATH = 'results/appendix'
relevant_columns = {
    'abby': [
        'backbone', 
        'sliding_style', 
        'sampler', 
        'mAP', 
        'f1_macro', 
        'acc', 
        'biting_f1_macro', 
        'biting_mAP',
        'biting_acc',
        'aggression_f1_macro',
        'aggression_mAP',
        'aggression_acc',
    ], 
    'mike': [
        'backbone',
        'sliding_style',
        'sampler',
        'mAP',
        'f1_macro',
        'acc',
        'movement_f1_macro',
        'movement_mAP',
        # 'movement_acc',
        'movement_precision_macro', 
        'movement_recall_macro',

        'biting_f1_macro',
        'biting_mAP',
        # 'biting_acc',
        'biting_precision_macro',
        'biting_recall_macro',

        'foraging_f1_macro',
        'foraging_mAP',
        # 'foraging_acc',
        'foraging_precision_macro',
        'foraging_recall_macro',

        'interactions_f1_macro',
        'interactions_mAP',
        # 'interactions_acc',
        'interactions_precision_macro',
        'interactions_recall_macro',

        'habitat_f1_macro',
        'habitat_mAP',
        # 'habitat_acc',
        'habitat_precision_macro',
        'habitat_recall_macro',

        'other_f1_macro',
        'other_mAP',
        # 'other_acc',
        'other_precision_macro',
        'other_recall_macro',
    ]
}
os.makedirs(OUTPUT_PATH, exist_ok=True)
for DATASET in DATASETS:
    for TOL in TOLS:
        PATH = f'results/subgroup_metrics/{DATASET}_eval_tol={TOL}_w_subgroup_metrics.csv'
        df = pd.read_csv(PATH)
        cleaned_df = df[relevant_columns[DATASET]].copy()
        cleaned_df.to_csv(os.path.join(OUTPUT_PATH, f"{DATASET}_tol={TOL}_appendix.csv"), index=False)