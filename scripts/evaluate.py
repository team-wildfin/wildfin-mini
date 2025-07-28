'''
run oriented evaluation
'''
import os 
import yaml
import wandb
from submission import get_slurm_submission_command
import subprocess
from datetime import datetime, timezone

# ==== CONFIG ====
ENTITY = "fish-benchmark"
PROJECT = "coralcam"
PARALLEL = False
# cutoff = datetime(2024, 5, 12, 17, 44, tzinfo=timezone.utc)
#srun -p gpu -n 8 --mem=48g --time=24:00:00 --pty /bin/bash
filt = {
    "dataset": "*",
    "sliding_style": "*", #"sliding_window_w_temp, sliding_window_w_stride",
    "backbone": "videomae, dino_large, resnet50", #dino, dino_large, videomae
    "pooling": "mean",
    "classifier": "mlp",
    "sampler": "balanced",
    "fulltune": "True, None" #True, False, None
}

def match_config(config, filt):
    def parse(x):
        x = x.strip().lower()
        if x == "none":
            return None
        if x == "true":
            return True
        if x == "false":
            return False
        return x

    return all(
        config.get(k) in [parse(x) for x in v.split(',')]
        or (v == '*' and k in config)
        for k, v in filt.items()
    )

def get_wrap_cmd(entity, project, run_id):
    return (
        f'python evaluation/main.py '
        f'--entity {entity} --project {project} --run {run_id} '
    )

def get_group_key(config):
    '''
    assumes config contains the keys in filt, otherwise it would've been filtered out
    '''

    return tuple(config.get(k, False) for k in filt.keys()) #Only possible None value now is for 'fulltune', which is a bool. 
#TODO: make this more robust to changes in filt

def main():
    api = wandb.Api()
    runs = api.runs(f'{ENTITY}/{PROJECT}', filters={"state": "finished"})
    filtered_runs = filter(lambda run: match_config(run.config, filt), runs)
    latest_runs_by_group = {}
    for run in filtered_runs:
        key = get_group_key(run.config)
        if key not in latest_runs_by_group or run.created_at > latest_runs_by_group[key].created_at:
            latest_runs_by_group[key] = run
    
    #sort by jey alphabetical order
    latest_runs_by_group = dict(sorted(latest_runs_by_group.items()))
    for key, run in latest_runs_by_group.items():
        print(key)
        print(run.id)
        print(run.created_at)

    
    for run in latest_runs_by_group.values():
        wrap_cmd = get_wrap_cmd(ENTITY, PROJECT, run.id)
        cmd = (get_slurm_submission_command(
                f"{run.id}", 
                os.path.join('logs', 'test', run.id), 
                wrap_cmd, 
                gpu_count=1)
               if PARALLEL else wrap_cmd)
        print(f"Running command for {run.id} with command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
if __name__ == "__main__":
    main()

# python evaluation/main.py --entity fish-benchmark --project abby --run q6150o9n