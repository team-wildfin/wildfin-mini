print("importing matcher tests...")
import unittest
print("importing experiments...")
from config.experiments.cvpr import CVPR_EXPS   
print("importing matcher...")
from scripts.matcher import WandbRunMatcher

class MatcherTest(unittest.TestCase):
    def setUp(self):
        entity = "fish-benchmark"
        project = "coralcam_eval"
        self.matcher = WandbRunMatcher(entity, project)
        self.experiments = list(
            filter(
                lambda exp: exp.dataset == "coralcam",
                CVPR_EXPS
            )
        )
    def test_match(self):
        matched = self.matcher.match(self.experiments)
        self.assertIsInstance(matched, dict)
        print(f"Matched {len(matched)} runs.")
        for exp_id, run_id in matched.items():
            self.assertIn(exp_id, [exp.id for exp in self.experiments])
            self.assertIsInstance(run_id, str)

if __name__ == "__main__":
    print("Running Matcher Tests")
    unittest.main()