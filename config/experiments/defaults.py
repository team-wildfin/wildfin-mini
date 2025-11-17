DEFAULT_FIELDS = {
    'pooling': 'mean',
    'classifier': 'mlp',
    'monitor': 'val_mAP',
    'learning_rate': 0.00005,
    'batch_size': 32,
    'weight_decay': 0.001,
    'shuffle': False,
    'optimizer': 'adam',
    'label_type': 'onehot',
    'max_samples_per_class': 1000,
    'freeze_backbone': False
}