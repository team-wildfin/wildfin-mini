import wandb
def log_best_model(checkpoint_callback, run):
    if checkpoint_callback.best_model_path:
        artifact = wandb.Artifact(
            name=f"model-{run.id}",
            type="model",
            metadata={
                "tags": run.tags,
                "config": dict(run.config),
                "notes": run.notes
            }
        )
        artifact.add_file(checkpoint_callback.best_model_path)
        run.log_artifact(artifact)

def log_latest_model(checkpoint_callback, run):
    if checkpoint_callback.last_model_path:
        artifact = wandb.Artifact(
            name=f"latest-model-{run.id}",
            type="model",
            metadata={
                "tags": run.tags,
                "config": dict(run.config),
                "notes": run.notes
            }
        )
        artifact.add_file(checkpoint_callback.last_model_path)
        run.log_artifact(artifact)

def log_dataset_summary(dataset, run):
    #store summary
    summary = dataset.get_summary()
    artifact = wandb.Artifact(
        name=f"dataset-{run.id}",
        type="dataset",
        metadata={
            "tags": run.tags,
            "config": dict(run.config),
            "notes": run.notes
        }
    )
    artifact.add_file(summary)
    run.log_artifact(artifact)