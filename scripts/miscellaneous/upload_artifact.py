import wandb

# Resume the specific run (or start a new one)
run = wandb.init(
    entity="fish-benchmark", 
    project="mike_eval",
    id="z8fnlcwz",  # or leave it out for a new run
    resume="must"       # use "must" to resume or raise error if run doesn't exist
)


# Create a new artifact
artifact = wandb.Artifact(
    name=f"test_metrics_{run.id}",  # unique name
    type="metrics",                # e.g., 'model', 'dataset', 'results'
    description="Manually uploaded file"
)

# Add file(s)
artifact.add_file("logs/test_metrics/z8fnlcwz.json")

# Log the artifact to this run
run.log_artifact(artifact)
run.finish()
