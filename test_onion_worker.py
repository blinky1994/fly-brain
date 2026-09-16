"""State-machine tests for outcome attribution, caching, and frozen evaluation."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from malecns_onion_brain import OnionBrain
from test_onion_outcome import OutcomeTests


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.brain = OnionBrain.__new__(OnionBrain)
        b = self.brain
        b.assay = SimpleNamespace(original=np.ones(3,dtype=np.float32),
            positions=np.arange(3),last_eligibility=np.ones(3,dtype=np.float32),
            brain=SimpleNamespace(weights=SimpleNamespace(data=np.ones(3,dtype=np.float32))),
            trial=Mock(return_value={"MBON11_spikes":6}))
        b.saved = {"trained":b.assay.original.copy()}
        b.training_dir = Path(self.tmp.name)
        b.training_rounds = b.outcome_id = 0
        b.response_cache = {}
        b.pending_outcome = None
        b.rng = np.random.default_rng(19)
        b.save_training = Mock()
        fixture = OutcomeTests()
        fixture.setUp()
        self.metrics = fixture.good

    def test_cache_invalidates_after_reward_and_duplicate_is_rejected(self):
        b = self.brain
        first = b.outcome_decide(0,explore=1)
        with self.assertRaises(ValueError):
            b.outcome_decide(0)
        with self.assertRaises(ValueError):
            b.observe_outcome(first["trial_id"]+1,self.metrics)
        b.observe_outcome(first["trial_id"],self.metrics,learn=False)
        second = b.outcome_decide(0,explore=1)
        self.assertTrue(second["cached_response"])
        result = b.observe_outcome(second["trial_id"],self.metrics)
        self.assertEqual(result["connections_changed"],3)
        b.save_training.assert_called_once()
        with self.assertRaises(ValueError):
            b.observe_outcome(second["trial_id"],self.metrics)
        third = b.outcome_decide(0,explore=0)
        self.assertFalse(third["cached_response"])
        self.assertEqual(b.assay.trial.call_count,2)

    def test_frozen_preserves_weights_and_no_cache_recomputes(self):
        b = self.brain
        original = b.saved["trained"].copy()
        for _ in range(2):
            d = b.outcome_decide(0,explore=1,use_cache=False)
            b.observe_outcome(d["trial_id"],self.metrics,learn=False)
        np.testing.assert_array_equal(b.saved["trained"],original)
        self.assertEqual(b.assay.trial.call_count,2)
        b.save_training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
