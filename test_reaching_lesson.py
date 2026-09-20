import unittest
import numpy as np
from reaching_lesson import ReachingLesson


class ReachingTests(unittest.TestCase):
    def test_passive_body_cannot_pass_and_target_stays_fixed(self):
        scene = ReachingLesson()
        target = scene.target.copy()
        self.assertAlmostEqual(scene.start_distance,.1)
        self.assertFalse(scene.cutting_enabled)
        for _ in range(100):
            scene.step(np.zeros(14))
        self.assertFalse(scene.metrics()['reach_success'])
        self.assertEqual(scene.metrics()['cuts'],0)
        scene.reset(.006)
        np.testing.assert_array_equal(scene.target,target)
        self.assertEqual(scene.dwell,0)
        self.assertFalse(scene.metrics()['reach_success'])

    def test_transient_entry_is_not_success(self):
        scene = ReachingLesson()
        # Probe the dwell gate independently of whether the policy can reach.
        scene.distance = lambda: 0.
        for _ in range(10): scene.step(np.zeros(14))
        self.assertTrue(scene.metrics()['reach_success'])
        scene.distance = lambda: .1
        scene.step(np.zeros(14))
        self.assertFalse(scene.metrics()['reach_success'])


if __name__ == '__main__': unittest.main()
