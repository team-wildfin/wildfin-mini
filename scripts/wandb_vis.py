import wandb
import pandas as pd
import argparse

# from matplotlib import pyplot as plt
# from matplotlib.ticker import FuncFormatter

# custom_labels = {
#     "otqfq10b": "DINO (stride=1, TI=1)",
#     "ldroyyac": "VideoMAE (stride=1, TI=1)"
# }

# api = wandb.Api()
# plt.figure(figsize=(8, 5))
# metric = "val_mAP"
# title = "Validation mAP curve"

# for run_id in custom_labels.keys():
#     run = api.run(f"fish-benchmark/abby/{run_id}")
#     history = run.history()
#     label = custom_labels.get(run_id, run.name)
    
#     steps = history["trainer/global_step"]
#      # Fill missing values or drop them
#     if history[metric].isnull().any():
#         history[metric] = history[metric].interpolate(method='linear', limit_direction='both')


#     plt.plot(steps, history[metric], label=label)

# # Format x-axis ticks like "1k", "2k", etc.
# def thousands_formatter(x, _):
#     if x >= 1000:
#         return f"{int(x/1000)}k"
#     return str(int(x))

# plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

# # Start axes at 0
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.xlabel("Step")
# plt.ylabel(metric)
# plt.title(title)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # Save as vector image (PDF or SVG)
# plt.savefig(f"figures/{metric}.pdf")



def upload_csv_to_wandb(csv_path, project, entity, table_name="results_table", run_name=None):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Initialize wandb run
    wandb.init(
        project=project,
        entity=entity,
        name=run_name or f"upload-{table_name}",
        job_type="upload_csv"
    )

    # Create W&B table
    table = wandb.Table(dataframe=df)

    # Log table to wandb
    wandb.log({table_name: table})

    print(f"Uploaded CSV '{csv_path}' as W&B table '{table_name}'")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/fishfollow_results_label_tolerance_0.csv", required=False, help="Path to CSV file")
    parser.add_argument("--project", type=str, default="fishfollow_eval", required=False, help="W&B project name")
    parser.add_argument("--entity", type=str, default="fish-benchmark", required=False, help="W&B entity name")
    parser.add_argument("--table_name", type=str, default="results_table_label_tolerance_0", help="Name for the W&B table")
    parser.add_argument("--run_name", type=str, default=None, help="Optional name for the W&B run")
    args = parser.parse_args()

    upload_csv_to_wandb(
        csv_path=args.csv,
        project=args.project,
        entity=args.entity,
        table_name=args.table_name,
        run_name=args.run_name,
    )
