"""Reward integrity and direction-of-learning checks without the large brain."""
import unittest
import numpy as np
from onion_outcome import cutting_reward, update_weights


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.good = dict(seam_broken=True,first_pad_contact_s=0.5,cut_time_s=1.3,
            pad_force_at_fracture_model_units=1.8,prefracture_slip_mm=0.05,
            peak_blade_force_model_units=26,knife_pad_force_model_units=0)

    def test_reward_requires_cut_and_hold(self):
        self.assertEqual(cutting_reward("chop",self.good)[0],1)
        for changes in [dict(seam_broken=False),dict(first_pad_contact_s=None),
                        dict(pad_force_at_fracture_model_units=0),dict(prefracture_slip_mm=0.2),
                        dict(peak_blade_force_model_units=60),dict(knife_pad_force_model_units=1)]:
            self.assertEqual(cutting_reward("chop",dict(self.good,**changes))[0],-1)
        self.assertEqual(cutting_reward("hold",self.good)[0],0)

    def test_action_credit_and_opposite_updates(self):
        old = np.ones(3,dtype=np.float32)
        trace = np.array([0,1,10])
        positive = update_weights(old,old,trace,1,"chop")
        negative = update_weights(old,old,trace,-1,"chop")
        self.assertEqual(positive[0],1)
        self.assertTrue(np.all(positive[1:] < old[1:]))
        self.assertTrue(np.all(negative[1:] > old[1:]))
        np.testing.assert_array_equal(update_weights(old,old,trace,1,"hold"),old)
        self.assertTrue(np.all(positive >= 0.1))
        self.assertTrue(np.all(negative <= 2))


if __name__ == "__main__":
    unittest.main()
