from typing import Optional, List, Dict, Callable, Iterable
from fish_benchmark.typing.experiment import Experiment
import wandb
import logging
# Configure the *root logger* so it handles logs from any module
logging.basicConfig(
    level=logging.INFO,  # or INFO if you want less verbosity
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger()
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
        for run in source_runs:
            if not self._has_any_artifact(run):
                continue
            for exp in experiments:
                #u0r08iio
                if self.match_config(run.config, exp):
                    if exp.id not in matched_runs or run.created_at > matched_runs[exp.id].created_at:
                        matched_runs[exp.id] = run
                    break
        logger.info(f"Found {len(matched_runs)} matching runs in {self.source_project} for {len(experiments)} experiments.")
        return {k: v.id for k, v in matched_runs.items()}

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
                logger.info(f"Config mismatch: {k} -> {run_val} != {v} in {reference.id}")
                return False
        return True