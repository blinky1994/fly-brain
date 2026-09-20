"""Physics and success gates, independent of whether neural training succeeds."""
import unittest
import mujoco as mj
import numpy as np
from bench_press import BenchPress


class BenchTests(unittest.TestCase):
    def setUp(self): self.scene=BenchPress()

    def test_grips_are_on_distal_feet_not_proximal_joints(self):
        s=self.scene
        for side in ['lf','rf']:
            tip=s.model.site(f'{side}_tip').id
            pad=s.model.geom(f'{side}_press_pad').id
            distal=s.model.body(f'lifter/{side}_tarsus5').id
            self.assertEqual(s.model.site_bodyid[tip],distal)
            self.assertEqual(s.model.geom_bodyid[pad],distal)
            np.testing.assert_allclose(s.data.site_xpos[tip],s.data.geom_xpos[pad])
            self.assertGreater(np.linalg.norm(s.data.site_xpos[tip]-
                s.data.xpos[s.model.body(f'lifter/{side}_tarsus1').id]),.4)
        self.assertLess(s.grip_errors().max(),1e-8)

    def test_passive_body_does_not_lift(self):
        s=self.scene
        for _ in range(100): s.step(np.zeros(14))
        self.assertLess(s.max_height,.005)
        self.assertFalse(s.metrics()['lift_success'])
        self.assertNotIn(s.model.joint('bar_slide').id,s.model.actuator_trnid[:,0])
        self.assertEqual(len(s.observation()),22)

    def test_secured_legs_can_lift_passive_bar(self):
        s=self.scene
        # Mechanical fixture only: find a small upward leg displacement using
        # the kinematic Jacobian. The trainer never calls this or uses IK.
        target=s.data.qpos[s.motor_qadr].copy()
        for side,indices in [('lf',np.arange(7)),('rf',np.arange(7,14))]:
            jac=np.zeros((3,s.model.nv)); rot=np.zeros_like(jac)
            mj.mj_jacSite(s.model,s.data,jac,rot,s.model.site(f'{side}_tip').id)
            joints=s.model.actuator_trnid[s.motor_indices[indices],0]
            gradient=jac[2,s.model.jnt_dofadr[joints]]
            target[indices] += .15*gradient/max(np.linalg.norm(gradient),1e-9)
        s.data.ctrl[s.motor_indices]=target
        for _ in range(25): s.step(np.zeros(14))
        self.assertGreater(s.max_height,.02)
        self.assertLess(np.max(s.grip_errors()),.02)

    def test_airborne_bar_without_bilateral_contact_is_not_success(self):
        s=self.scene
        s.data.eq_active[s.grip_ids]=False
        s.data.qpos[s.bar_qadr]=.3
        mj.mj_forward(s.model,s.data)
        s.step(np.zeros(14))
        self.assertFalse(s.metrics()['lift_success'])
        s.reset()
        self.assertEqual(s.max_height,0)
        self.assertEqual(s.hold,0)


if __name__=='__main__': unittest.main()
