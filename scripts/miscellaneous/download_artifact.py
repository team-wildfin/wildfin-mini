import wandb
api = wandb.Api()
run = api.run("fish-benchmark/mike_eval/z8fnlcwz")

print("Artifacts:")
for artifact in run.logged_artifacts():
    print(f"- {artifact.name} ({artifact.type})")