"""Geometry and state invariants, separate from any claim of learned slicing."""
import unittest
import numpy as np
from intact_onion import IntactOnion, TOTAL_MASS


class IntactOnionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene=IntactOnion()

    def setUp(self):
        self.scene.reset()

    def test_starts_as_one_physical_body_and_does_not_cut_at_rest(self):
        s=self.scene
        self.assertEqual(list(s.pieces),['whole_onion'])
        self.assertEqual(s.model.neq,0)
        self.assertFalse(any('whole_onion_front' in (s.model.geom(i).name or '') for i in range(s.model.ngeom)))
        for _ in range(15): s.step(np.zeros(14))
        self.assertEqual(s.metrics()['cuts'],0)
        self.assertEqual(s.metrics()['physical_pieces'],1)

    def test_arbitrary_split_preserves_state_and_mass(self):
        s=self.scene
        # Exercise the topology operation directly, not evidence of physical cutting.
        q=s.data.qpos[s.motor_qadr].copy()
        body=s.model.body('whole_onion').id
        origin=s.data.xpos[body].copy()
        joint=s.model.joint('whole_onion_free').id
        dadr=s.model.jnt_dofadr[joint]
        s.data.qvel[dadr:dadr+6]=[.01,.02,.03,.04,.05,.06]
        s._split('whole_onion',-.2137)
        self.assertEqual(len(s.pieces),2)
        self.assertAlmostEqual(sum(p['mass'] for p in s.pieces.values()),TOTAL_MASS)
        np.testing.assert_array_equal(s.data.qpos[s.motor_qadr],q)
        for name,piece in s.pieces.items():
            np.testing.assert_allclose(s.data.xpos[s.model.body(name).id],origin)
            joint=s.model.joint(name+'_free').id
            dadr=s.model.jnt_dofadr[joint]
            np.testing.assert_allclose(s.data.qvel[dadr:dadr+6],[.01,.02,.03,.04,.05,.06])
            self.assertTrue(piece['lo']==-.2137 or piece['hi']==-.2137)
        self.assertEqual(s.metrics()['cuts'],0)  # Internal operation earns no credit.
        s.reset()
        self.assertEqual(s.metrics()['physical_pieces'],1)


if __name__=='__main__':
    unittest.main()
