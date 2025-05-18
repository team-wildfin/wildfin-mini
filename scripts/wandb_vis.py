import wandb
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

custom_labels = {
    "otqfq10b": "DINO (stride=1, TI=1)",
    "ldroyyac": "VideoMAE (stride=1, TI=1)"
}

api = wandb.Api()
plt.figure(figsize=(8, 5))
metric = "train_loss"
title = "Train Loss curve"

for run_id in custom_labels.keys():
    run = api.run(f"fish-benchmark/abby/{run_id}")
    history = run.history()
    label = custom_labels.get(run_id, run.name)
    
    steps = history["trainer/global_step"]
     # Fill missing values or drop them
    if history[metric].isnull().any():
        history[metric] = history[metric].interpolate(method='linear', limit_direction='both')


    plt.plot(steps, history[metric], label=label)

# Format x-axis ticks like "1k", "2k", etc.
def thousands_formatter(x, _):
    if x >= 1000:
        return f"{int(x/1000)}k"
    return str(int(x))

plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

# Start axes at 0
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel("Step")
plt.ylabel(metric)
plt.title(title)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save as vector image (PDF or SVG)
plt.savefig(f"figures/{metric}.svg")