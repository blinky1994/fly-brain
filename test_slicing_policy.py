import tempfile
import unittest
from pathlib import Path
import numpy as np
from slicing_policy import SlicingPolicy,PROFILES,slice_reward


class SlicingPolicyTests(unittest.TestCase):
    def metrics(self,clean=5,detached=5):
        return dict(target_slices=5,clean_slices=clean,slices_detached=detached,
                    peak_blade_force=15,blade_foot_force=0,max_remainder_slip_mm=.01)

    def test_clean_completion_outscores_partial_detachment(self):
        self.assertGreater(slice_reward(self.metrics()),slice_reward(self.metrics(0)))
        bad=self.metrics();bad['blade_foot_force']=.1
        self.assertEqual(slice_reward(bad),-1)
        bad=self.metrics();bad['failure_reason']='unreachable remainder'
        self.assertEqual(slice_reward(bad),-1)
        with self.assertRaises(ValueError):
            slice_reward(self.metrics(6))

    def test_learning_and_checkpoint_resume(self):
        policy=SlicingPolicy()
        self.assertEqual(policy.choose(False)[0],0)
        for i in range(len(PROFILES)):
            action,_=policy.choose(True)
            self.assertEqual(action,i)
            policy.update(action,.9 if i==2 else -.2)
        self.assertEqual(policy.choose(False)[0],2)
        previous=policy.counts.copy()
        policy.choose(False)
        np.testing.assert_array_equal(policy.counts,previous)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'latest.json'; policy.save(path)
            resumed=SlicingPolicy(path)
            np.testing.assert_array_equal(resumed.counts,policy.counts)
            np.testing.assert_array_equal(resumed.values,policy.values)
            self.assertEqual([resumed.choose(True) for _ in range(10)],
                             [policy.choose(True) for _ in range(10)])
            with self.assertRaises(ValueError):
                SlicingPolicy(path,slices=5)


if __name__=='__main__':
    unittest.main()
