import tempfile
import time
from pathlib import Path
import unittest
import numpy as np
from neuron_activity import ActivityState, spike_frame


class NeuronActivityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'atlas.npz'
        np.savez(self.path,schema=1,body_ids=[10,20,30],xyz=[[0,0,0],[1,2,3],[np.nan]*3],
                 types=['A','B','C'],classes=['cb_intrinsic','vnc_motor','vnc_sensory'],
                 sides=['L','R','?'],motor=[False,True,False],sensor=[False,False,True])
        self.state=ActivityState(self.path)

    def tearDown(self): self.temp.cleanup()

    def test_frames_are_matched_by_id_and_replace_activity(self):
        self.state.update(spike_frame([30,10],[2,1],20))
        np.testing.assert_array_equal(self.state.counts,[1,0,2])
        self.assertEqual(self.state.counts[~self.state.positioned].sum(),2)
        self.state.update(spike_frame([20],[3],40))
        np.testing.assert_array_equal(self.state.counts,[0,3,0])
        self.state.clear()
        self.assertFalse(self.state.has_frame)
        self.assertEqual(self.state.counts.sum(),0)

    def test_invalid_or_duplicate_ids_are_rejected(self):
        for pairs in [[[99,1]],[[10,1],[10,2]],[[10,-1]],[[10,.5]]]:
            with self.assertRaises(ValueError):
                self.state.update(dict(neuron_spikes=pairs,neural_time_ms=20,window_ms=20))

    def test_empty_frame_clears_spikes(self):
        self.state.update(spike_frame([10],[1],20))
        self.state.update(spike_frame([10],[0],40))
        self.assertEqual(self.state.counts.sum(),0)

    def test_renderer_clears_stale_glow_and_picks_only_visible_cells(self):
        from neuron_viewer import NeuronRenderer
        r=NeuronRenderer(self.path)
        empty=r.render(640,480)
        r.update(spike_frame([20],[3],20))
        lit=r.render(640,480)
        self.assertFalse(np.array_equal(empty[130:385],lit[130:385]))
        r.last_frame=time.monotonic()-4
        stale=r.render(640,480)
        np.testing.assert_array_equal(empty[130:385],stale[130:385])
        r.projection=(np.array([100,100,100]),np.array([200,200,200]),np.array([False,True,False]))
        r.pick(100,200)
        self.assertEqual(r.selected,1)


if __name__=='__main__': unittest.main()
