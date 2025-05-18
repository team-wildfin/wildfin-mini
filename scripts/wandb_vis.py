import wandb
from matplotlib import pyplot as plt

custom_labels = {
    "otqfq10b": "DINO (stride=1, TI=1)",
    "ldroyyac": "VideoMAE (stride=1, TI=1)"
}

api = wandb.Api()
plt.figure(figsize=(8, 5))

for run_id in custom_labels.keys():
    run = api.run(f"fish-benchmark/abby/{run_id}")
    history = run.history(samples=10000)
    label = custom_labels.get(run_id, run.name)
    
    steps = history["trainer/global_step"]
    train_loss = history["train_loss"]  # Make sure this key exists in your run

    plt.plot(steps, train_loss, label=label)

plt.xlabel("Step")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save as vector image (PDF or SVG)
plt.savefig("figures/train_loss.pdf")
# Optional: plt.savefig("figures/train_loss.png", dpi=300)
