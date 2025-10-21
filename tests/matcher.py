import logging 
from fish_benchmark.utils.general import setup_logger
logger = setup_logger(
    "matcher_test",
    console=True,
    file=False,
    level=logging.DEBUG
)
logger.debug("importing matcher tests...")
import unittest
logger.debug("importing experiments...")
from config.experiments.cvpr import CVPR_EXPS   
logger.debug("importing matcher...")
from scripts.matcher import WandbRunMatcher
from fish_benchmark.typing.experiment import Experiment

class MatcherTest(unittest.TestCase):
    def setUp(self):
        entity = "fish-benchmark"
        project = "coralcam_eval"
        self.matcher = WandbRunMatcher(entity, project)
        # self.experiments = list(
        #     filter(
        #         lambda exp: exp.dataset == "coralcam",
        #         CVPR_EXPS
        #     )
        # )
        self.experiments = [
            Experiment
        ]
    def test_match(self):
        matched = self.matcher.match(self.experiments)
        self.assertIsInstance(matched, dict)
        print(f"Matched {len(matched)} runs.")
        for exp_id, run_id in matched.items():
            self.assertIn(exp_id, [exp.id for exp in self.experiments])
            self.assertIsInstance(run_id, str)

if __name__ == "__main__":
    logger.info("Running Matcher Tests")
    unittest.main()