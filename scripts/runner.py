from typing import Optional, List, Dict, Callable
from fish_benchmark.types import Experiment
import wandb
import logging
logger = logging.getLogger(__name__)
class WandbRunner: 
    def __init__(self, entity: str, source_project: str, dest_project: Optional[str]): 
        self.entity = entity
        self.source_project = source_project
        self.dest_project = dest_project 
        self.api = wandb.Api()

    def _get_completed_run_ids(self, experiments: List[Experiment]) -> set[str]: 
        assert self.dest_project is not None, "Destination project must be set for completed run ID retrieval."
        runs = self.api.runs(f"{self.entity}/{self.dest_project}", filters={"state": "finished"})
        already_evaluated = set()
        for run in runs:
            for exp in experiments:
                if self.match_config(run.config, exp):
                    already_evaluated.add(exp.id)
                    break
        logger.info(f"Already evaluated runs: {len(already_evaluated)}")
        return already_evaluated

    def get_matched_run_ids(self, experiments: List[Experiment], non_duplicate = False) -> Dict[str, str]:
        """
        Get matched run IDs for the given experiments.
        If non_duplicate is True, it will filter out runs that have already been evaluated.
        """
        matched_runs = {}
        source_runs = self.api.runs(f"{self.entity}/{self.source_project}", filters={"state": "finished"})
        for run in source_runs:
            for exp in experiments:
                if self.match_config(run.config, exp):
                    if exp.id not in matched_runs or run.created_at > matched_runs[exp.id].created_at:
                        matched_runs[exp.id] = run
                    break
        logger.info(f"Found {len(matched_runs)} matching runs in {self.source_project} for {len(experiments)} experiments.")
        if non_duplicate: 
            already_evaluated = self._get_completed_run_ids(experiments)
            matched_runs = {k: v for k, v in matched_runs.items() if k not in already_evaluated}
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
                logger.debug(f"Config mismatch: {k} -> {run_val} != {v} in {reference.id}")
                return False
        return True