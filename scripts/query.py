from .matcher import WandbRunMatcher
from typing import List, Dict
import pprint
import logging
from fish_benchmark.typing.experiment import Experiment

def query_pending_experiments(runner: WandbRunMatcher, experiments: List[Experiment]) -> List[Experiment]:
    matched_runs = runner.match(experiments)     # {exp_id: run_id}
    pending_exps = [exp for exp in experiments if len(matched_runs[exp.id]) == 0]
    return pending_exps

def query_trained(train_matcher: WandbRunMatcher, experiments: List[Experiment]) -> Dict[str, List[str]]:
    '''
    Given a list of experiments, return a dict of trained experiments {exp_id: [train_run_ids]}
    '''
    matched_runs = train_matcher.match(experiments)     # {exp_id: run_id}
    trained = {exp_id: v for exp_id, v in matched_runs.items() if len(v) > 0}
    return trained

def query_evaluated(eval_matcher: WandbRunMatcher, 
                    trained: Dict[str, List[str]]) -> Dict[str, List[str]]:
    '''
    Given a dict of trained experiments {exp_id: [train_run_ids]}, return a dict of evaluated experiments
    '''
    evaluation_status = {
        exp_id: eval_matcher.match_by_train_id(train_ids)
        for exp_id, train_ids in trained.items()
    }
    evaluated = {exp_id: v for exp_id, v in evaluation_status.items() if len(v) > 0}
    return evaluated

def query_pending_evaluations(train_matcher: WandbRunMatcher, 
                              eval_matcher: WandbRunMatcher, 
                              experiments: list[Experiment]) -> dict[str, str]:
    trained = query_trained(train_matcher, experiments)
    evaluated = query_evaluated(eval_matcher, trained)
    pending_eval = {exp_id: trained[exp_id][0] for exp_id in set(trained.keys()) - set(evaluated.keys())}
    logger.info(f"Pending evaluations for {len(pending_eval)} experiments:")
    logger.info(pprint.pprint(pending_eval))
    return pending_eval
