from typing import Optional, List, Dict, Callable, Iterable
from fish_benchmark.typing.experiment import Experiment
import wandb
import logging
# Configure the *root logger* so it handles logs from any module
logger = logging.getLogger(__name__)
class WandbRunMatcher: 
    '''
    A general class to run any function using wandb runs from the source project and output to the destination project.
    It can detect if the runs 
    '''
    def __init__(self, entity: str, project: str): 
        self.entity = entity
        self.source_project = project
        self.api = wandb.Api()

    def _has_any_artifact(self, run) -> bool:
        """
        Returns True if the run has at least one logged artifact.
        """
        try:
            return len(run.logged_artifacts()) > 0
        except Exception as e:
            logger.debug(f"Error checking artifacts for run {run.id}: {e}")
            return False

    def match(self, experiments: List[Experiment]) -> Dict[str, str]:
        """
        Get matched run IDs for the given experiments.
        If non_duplicate is True, it will filter out runs that have already been evaluated.
        """
        matched_runs = {}
        source_runs = self.api.runs(f"{self.entity}/{self.source_project}", filters={"state": "finished"})
        logger.debug(f"Found {len(source_runs)} finished runs in {self.source_project}.")
        print(f"Found {len(source_runs)} finished runs in {self.source_project}.")
        source_runs = [run for run in source_runs if self._has_any_artifact(run)]
        for exp in experiments: 
            runs = []
            for run in source_runs: 
                if self.match_config(run.config, exp):
                    runs.append(run.id)
            matched_runs[exp.id] = runs
        return matched_runs

    def match_by_train_id(self, train_ids: List[str]) -> List[Optional[str]]: 
        source_runs = self.api.runs(f"{self.entity}/{self.source_project}", filters={"state": "finished"})
        matched_ids = []
        for train_id in train_ids:
            matched_id = None
            for run in source_runs:
                if run.config.get('training_run_id', None) == train_id:
                    matched_id = run.id
                    break
            matched_ids.append(matched_id)
        return matched_ids

    @staticmethod 
    def match_config(run_config: dict, reference: Experiment) -> bool:
        """Check if a W&B run config matches a TrainConfig, ignoring 'id' and defaulting missing fields."""
        def get(run_config, key):
            if key == 'fulltune':
                return run_config.get(key, False)
            elif key == 'label_type':
                return run_config.get(key, 'onehot')
            return run_config.get(key)

        run_config = {k.lower(): v for k, v in run_config.items()}
        ref_dict = reference.model_dump(mode="json", exclude={"id"})

        for k, v in ref_dict.items():
            run_val = get(run_config, k)
            if run_val != v:
                logger.debug(f"Config mismatch: {k} -> {run_val} != {v} in {reference.id}")
                return False
        return True