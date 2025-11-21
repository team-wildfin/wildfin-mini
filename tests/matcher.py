import logging 
from vision_bench.utils.general import setup_logger
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
from vision_bench.management.matcher import WandbRunMatcher
from vision_bench.typing.experiment import Experiment
from config.experiments.neurips import RESNET_FULLTUNE

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
        self.experiments = RESNET_FULLTUNE
    def test_match(self):
        matched = self.matcher.match(self.experiments)
        self.assertIsInstance(matched, dict)
        print(matched)
        for exp_id, run_id in matched.items():
            self.assertIn(exp_id, [exp.id for exp in self.experiments])
            self.assertIsInstance(run_id, str)

if __name__ == "__main__":
    logger.info("Running Matcher Tests")
    unittest.main()