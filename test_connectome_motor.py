"""Actual full-graph neural causality/frozen/save checks, not cooking benchmarks."""
import tempfile
import unittest
import numpy as np
from malecns_motor_brain import MotorBrain, ROOT


class ConnectomeMotorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names=[f'{side}axis{i}' for side in ['lf_','rf_'] for i in range(7)]
        cls.brain=MotorBrain(cls.names,22)

    def setUp(self):
        b=self.brain
        b.pending=None
        b.gains[:]=0
        b._apply(b.gains)
        b.best_reward=None

    def test_fresh_motor_spikes_drive_commands_and_ablation_stops_them(self):
        b=self.brain
        self.assertEqual(b.brain.n,166700)
        self.assertEqual(b.brain.graph.nnz,25582938)
        before=b.brain.weights.data[b.positions].copy()
        b.begin()
        actions=[b.act(np.zeros(22))['action'] for _ in range(10)]
        result=b.finish(0,reference=True)
        self.assertGreater(result['motor_spikes'],0)
        self.assertGreater(np.max(np.abs(actions)),0)
        self.assertEqual(result['retained_connections_changed'],0)
        np.testing.assert_array_equal(b.brain.weights.data[b.positions],before)
        b.begin(silence=True)
        muted=[b.act(np.zeros(22))['action'] for _ in range(10)]
        muted_result=b.finish(0)
        self.assertEqual(muted_result['total_spikes'],result['total_spikes'])
        np.testing.assert_array_equal(muted,np.zeros((10,14)))

    def test_internal_learning_checkpoint_and_frozen_evaluation(self):
        b=self.brain
        b.begin(); b.finish(0,reference=True)
        b.begin(learn=True)
        # Synthetic outcome exercises the update mechanism, not physical skill.
        result=b.finish(1)
        self.assertTrue(result['accepted'])
        self.assertGreater(result['retained_connections_changed'],0)
        before=b.brain.weights.data[b.positions].copy()
        self.assertTrue(np.array_equal(np.sign(before),np.sign(b.original)))
        b.begin(); result=b.finish(.5)
        self.assertEqual(result['retained_connections_changed'],0)
        self.assertEqual(result['candidate_connections_changed'],0)
        np.testing.assert_array_equal(before,b.brain.weights.data[b.positions])
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as folder:
            path=ROOT/folder/'motor.npz'
            b.save(path)
            resumed=MotorBrain(self.names,22,path)
            np.testing.assert_array_equal(resumed.brain.weights.data[resumed.positions],before)
            b.begin(); resumed.begin()
            np.testing.assert_array_equal(b.act(np.zeros(22))['action'],resumed.act(np.zeros(22))['action'])
            b.finish(0); resumed.finish(0)


if __name__=='__main__':
    unittest.main()
