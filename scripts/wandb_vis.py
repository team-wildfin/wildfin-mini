import wandb
from matplotlib import pyplot as plt

custom_labels = {
    "otqfq10b": "DINO (stride=1, TI=1)",
    "ldroyyac": "VideoMAE (stride=1, TI=1)"
}

api = wandb.Api()
for run_id in custom_labels.keys():
    run = api.run(f"fish-benchmark/abby/{run_id}")
    history = run.history(samples=10000)
    label = custom_labels.get(run_id, run.name)
    print(history.keys())
    plt.plot(history["val_mAP"], label=label)

plt.xlabel("Step")
plt.ylabel("Validation mAP")
plt.title("Validation mAP Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save as vector image (PDF or SVG)
plt.savefig("figures/validation_mAP.pdf")  # or use "train_loss.svg"
# Optional: save PNG too if needed
# plt.savefig("figures/train_loss.png", dpi=300)
