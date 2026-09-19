"""Outcome-trained motor-profile selection, separate from MaleCNS synapses."""
import json
from pathlib import Path
import numpy as np


PROFILES = [
    dict(name="offset fast",aim_bias=-.04,descent=.3,press=.005),
    dict(name="compensated steady",aim_bias=.025,descent=.7,press=.008),
    dict(name="compensated gentle",aim_bias=.025,descent=1.,press=.012),
    dict(name="light hold",aim_bias=.025,descent=.7,press=0),
    dict(name="inward aim",aim_bias=-.025,descent=.7,press=.02),
    dict(name="compensated fast",aim_bias=.025,descent=.3,press=.035),
]


def slice_reward(metrics):
    if metrics.get("failure_reason") is not None:
        return -1.0
    target=metrics["target_slices"]
    values=[metrics[k] for k in ["clean_slices","slices_detached","peak_blade_force",
                                "blade_foot_force","max_remainder_slip_mm"]]
    if target <= 0 or not np.isfinite(values).all() or min(values)<0:
        raise ValueError("Invalid slice measurements")
    if not 0 <= metrics["clean_slices"] <= metrics["slices_detached"] <= target:
        raise ValueError("Inconsistent slice counts")
    if metrics["blade_foot_force"] > .001:
        return -1.0
    # Completion dominates; partial detachment alone cannot score as a clean cut.
    score=(.85*metrics["clean_slices"]+.15*metrics["slices_detached"])/target
    score-=.2*min(1,metrics["max_remainder_slip_mm"]/.08)
    score-=.15*min(1,max(0,metrics["peak_blade_force"]-25)/25)
    return float(np.clip(score,-1,1))


class SlicingPolicy:
    def __init__(self,checkpoint=None,seed=29,slices=3):
        self.slices=slices
        self.counts=np.zeros(len(PROFILES),dtype=int)
        self.values=np.zeros(len(PROFILES))
        self.rng=np.random.default_rng(seed)
        if checkpoint:
            saved=json.loads(Path(checkpoint).read_text())
            if saved.get("schema") != 1 or saved["profiles"] != PROFILES:
                raise ValueError("Checkpoint belongs to a different motor-profile set")
            if saved.get("slices") != slices:
                raise ValueError("Use a fresh policy when changing the number of slices")
            self.counts=np.array(saved["counts"],dtype=int)
            self.values=np.array(saved["values"],dtype=float)
            if self.counts.shape!=(len(PROFILES),) or self.values.shape!=self.counts.shape:
                raise ValueError("Invalid policy shape")
            if (self.counts<0).any() or not np.isfinite(self.values).all():
                raise ValueError("Invalid policy state")
            self.rng.bit_generator.state=saved["rng_state"]

    def choose(self,learn=True):
        if learn:
            untried=np.flatnonzero(self.counts==0)
            if len(untried):
                return int(untried[0]),True
            if self.rng.random()<.2:
                return int(self.rng.integers(len(PROFILES))),True
        visited=np.flatnonzero(self.counts>0)
        return (int(visited[np.argmax(self.values[visited])]) if len(visited) else 0),False

    def update(self,action,reward):
        if not 0 <= action < len(PROFILES) or not np.isfinite(reward) or not -1 <= reward <= 1:
            raise ValueError("Invalid motor outcome")
        self.counts[action]+=1
        self.values[action]+=(reward-self.values[action])/self.counts[action]

    def save(self,path):
        path=Path(path)
        snapshot=dict(schema=1,slices=self.slices,profiles=PROFILES,counts=self.counts.tolist(),values=self.values.tolist(),
                      rng_state=self.rng.bit_generator.state,
                      scope="Motor-profile reward averages, not neural weights")
        temporary=path.with_suffix(".partial")
        temporary.write_text(json.dumps(snapshot,indent=2))
        temporary.replace(path)
