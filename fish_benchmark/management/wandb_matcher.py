
from typing import Optional, List, Dict, Callable, Iterable
from fish_benchmark.typing.experiment import Experiment
import wandb
import logging
from fish_benchmark.typing.types import RunState
from fish_benchmark.management.matcher import Matcher
import json
import os 
import glob
# Configure the *root logger* so it handles logs from any module
logger = logging.getLogger(__name__)

class WandbRunMatcher(Matcher): 
    '''
    A general class to run any function using wandb runs from the source project and output to the destination project.
    It can detect if the runs 
    '''
    def __init__(self, 
                 entity: str, 
                 project: str, 
                 local_artifact_dir: str = "./artifacts", 
                 default_values: Optional[Dict[str, any]] = {}): 
        """
        Args:
            entity: The W&B entity (user or team) to use for API calls.
            project: The W&B project to use for API calls.
            local_artifact_dir: The local directory to store artifacts. If an artifact is not found
                locally, it will be downloaded from W&B and stored here.
            default_values: A dictionary of default values for config keys. If a key is missing in
                the run config, it will be filled in with the value from this dictionary before matching.
        """
        self.entity = entity
        self.project = project
        self.local_artifact_dir = local_artifact_dir
        self.default_values = default_values

    def _has_any_artifact(self, run) -> bool:
        """
        Returns True if the run has at least one logged artifact.
        """
        try:
            return len(run.logged_artifacts()) > 0
        except Exception as e:
            logger.debug(f"Error checking artifacts for run {run.id}: {e}")
            return False

    def match(self, experiments: List[Experiment], states: List[RunState] = ["finished"]) -> Dict[str, str]:
        """
        Get matched run IDs for the given experiments.
        If non_duplicate is True, it will filter out runs that have already been evaluated.
        """
        matched_runs = {}
        source_runs = wandb.Api().runs(f"{self.entity}/{self.source_project}", filters={"state": {"$in": states}})
        logger.debug(f"Found {len(source_runs)} finished runs in {self.source_project}.")
        print(f"Found {len(source_runs)} finished runs in {self.source_project}.")
        #source_runs = [run for run in source_runs if self._has_any_artifact(run)]
        for exp in experiments: 
            runs = []
            for run in source_runs: 
                if self.match_config(run.config, exp):
                    runs.append(run.id)
            matched_runs[exp.id] = runs
        return matched_runs

    def match_by_train_id(self, train_ids: List[str], states: List[RunState] = ["finished"]) -> List[str]: 
        """
        Returns the set of evaluation run IDs that correspond to any of the given training run IDs.
        In most cases the train ids should be of the same experiment. 
        """
        source_runs = wandb.Api().runs(f"{self.entity}/{self.source_project}", filters={"state": {"$in": states}})
        matched_ids = []
        for train_id in train_ids:
            for run in source_runs:
                if run.config.get('training_run_id', None) == train_id:
                    matched_ids.append(run.id)
                    break
        return matched_ids
    
    def get_latest(self, run_ids: List[str]) -> str: 
        '''
        From the given run IDs, return the latest one based on the created_at timestamp.
        '''
        api = wandb.Api()
        latest_run = None
        latest_time = None
        for run_id in run_ids: 
            run = api.run(f"{self.entity}/{self.project}/{run_id}")
            if latest_time is None or run.created_at > latest_time: 
                latest_time = run.created_at
                latest_run = run_id
        return latest_run

    def match_config(self, run_config: dict, reference: Experiment) -> bool:
        """Check if a W&B run config matches a TrainConfig, ignoring 'id' and defaulting missing fields."""
        def get(run_config, key):
            if key in self.default_values:
                return run_config.get(key, self.default_values[key])
            return run_config.get(key)

        run_config = {k.lower(): v for k, v in run_config.items()}
        ref_dict = reference.model_dump(mode="json", exclude={"id"})

        for k, v in ref_dict.items():
            run_val = get(run_config, k)
            if run_val != v:
                logger.debug(f"Config mismatch: {k} -> {run_val} != {v} in {reference.id}")
                return False
        return True
    
    def get_run_config(self, run_id: str) -> Dict:
        api = wandb.Api()
        return api.run(f"{self.entity}/{self.project}/{run_id}").config
    

    def get_artifact(self,
                     local_path: str, 
                     remote_path: Optional[str] = None) -> str:
        """
        Get the artifact data for the given run ID and artifact name.
        local_path: The local path to the artifact file. The path is relative to the local_artifact_dir. If the file exists locally, it will be loaded from there.
        remote_path: The remote path to the artifact file in W&B.

        Returns: 

        Local artifacts and remote artifacts may have different naming conventions. 
        Priotitize local artifacts if they exist, otherwise download from W&B.
        """
        api = wandb.Api()
        if local_path:
            try: 
                local_file_path = os.path.join(self.local_artifact_dir, local_path)
                if os.path.exists(local_file_path):
                    return local_file_path
            except Exception as e: 
                logger.warning(f"Failed to load local artifact {local_path}: {e}")
        elif remote_path:
            try: 
                artifact_path = f"{self.entity}/{self.project}/{remote_path}"
                artifact = api.artifact(artifact_path)
                return artifact.download(root=self.local_artifact_dir)
            except Exception as e:
                logger.warning(f"Failed to download artifact {remote_path}: {e}")
        else: 
            raise FileNotFoundError(f"Failed to find artifact: {local_path} or {remote_path}")