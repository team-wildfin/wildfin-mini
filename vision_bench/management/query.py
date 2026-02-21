from vision_bench.management.matcher import Matcher
from typing import List, Dict
import pprint
import logging
from vision_bench.typing.experiment import Experiment

logger = logging.getLogger(__name__)

def query_pending_experiments(runner: Matcher, experiments: List[Experiment]) -> List[Experiment]:
    matched_runs = runner.match(experiments, states = ['pending', 'running', 'finished'])     # {exp_id: run_id}
    pending_exps = [exp for exp in experiments if len(matched_runs[exp.id]) == 0]
    return pending_exps

def query_trained(train_matcher: Matcher, experiments: List[Experiment]) -> Dict[str, List[str]]:
    '''
    Given a list of experiments, return a dict of trained experiments {exp_id: [train_run_ids]}
    '''
    matched_runs = train_matcher.match(experiments, states = ['finished'])     # {exp_id: run_id}
    trained = {exp_id: v for exp_id, v in matched_runs.items() if len(v) > 0}
    return trained

def query_evaluated(eval_matcher: Matcher, 
                    trained: Dict[str, List[str]]) -> Dict[str, List[str]]:
    '''
    Given a dict of trained experiments {exp_id: [train_run_ids]}, return a dict of evaluated experiments
    '''
    evaluation_status = {
        exp_id: eval_matcher.match_by_train_id(train_ids, states = ['finished', 'running'])
        for exp_id, train_ids in trained.items()
    }
    evaluated = {exp_id: v for exp_id, v in evaluation_status.items() if len(v) > 0}
    return evaluated

def query_pending_evaluations(train_matcher: Matcher, 
                              eval_matcher: Matcher, 
                              experiments: list[Experiment], 
                              rerun: bool = False) -> dict[str, str]:
    '''
    Given a list of experiments, return a dict of pending evaluations {exp_id: train_run_id}
    '''
    trained = query_trained(train_matcher, experiments)
    evaluated = query_evaluated(eval_matcher, trained) if not rerun else {}
    pending_eval = {exp_id: trained[exp_id][0] for exp_id in set(trained.keys()) - set(evaluated.keys())}
    logger.info(f"Pending evaluations for {len(pending_eval)} experiments:")
    logger.info(pprint.pprint(pending_eval))
    return pending_eval
