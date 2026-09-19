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
    dict(name="steady firm hold",aim_bias=.025,descent=.7,press=.016),
]
TRAIN_OFFSETS = [0,-.006,.006]


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
        self.context_counts=np.zeros((len(PROFILES),len(TRAIN_OFFSETS)),dtype=int)
        self.values=np.zeros(len(PROFILES))
        self.rng=np.random.default_rng(seed)
        if checkpoint:
            saved=json.loads(Path(checkpoint).read_text())
            if saved.get("schema") != 2:
                raise ValueError("Checkpoint lacks balanced-position history; start a fresh slicing session")
            previous_profiles=saved["profiles"]
            size=len(previous_profiles)
            if (not size or previous_profiles != PROFILES[:size]
                    or saved.get("training_offsets") != TRAIN_OFFSETS):
                raise ValueError("Checkpoint belongs to a different motor-profile set")
            if saved.get("slices") != slices:
                raise ValueError("Use a fresh policy when changing the number of slices")
            self.counts=np.array(saved["counts"],dtype=int)
            self.context_counts=np.array(saved["context_counts"],dtype=int)
            self.values=np.array(saved["values"],dtype=float)
            if self.counts.shape!=(size,) or self.values.shape!=self.counts.shape:
                raise ValueError("Invalid policy shape")
            if (self.counts<0).any() or not np.isfinite(self.values).all():
                raise ValueError("Invalid policy state")
            if (self.context_counts.shape!=(size,len(TRAIN_OFFSETS))
                    or (self.context_counts<0).any()
                    or not np.array_equal(self.context_counts.sum(axis=1),self.counts)):
                raise ValueError("Invalid training-position coverage")
            self.rng.bit_generator.state=saved["rng_state"]
            # New choices start untried; retain only outcomes for identical profiles.
            added=len(PROFILES)-size
            self.counts=np.pad(self.counts,(0,added))
            self.values=np.pad(self.values,(0,added))
            self.context_counts=np.pad(self.context_counts,((0,added),(0,0)))

    def choose(self,learn=True):
        if learn:
            untried=np.flatnonzero((self.context_counts==0).any(axis=1))
            if len(untried):
                return int(untried[np.argmin(self.counts[untried])]),True
            if self.rng.random()<.2:
                return int(self.rng.integers(len(PROFILES))),True
        visited=np.flatnonzero(self.counts>0)
        return (int(visited[np.argmax(self.values[visited])]) if len(visited) else 0),False

    def context_index(self,action):
        return int(np.argmin(self.context_counts[action]))

    def update(self,action,reward,context=None):
        if not 0 <= action < len(PROFILES) or not np.isfinite(reward) or not -1 <= reward <= 1:
            raise ValueError("Invalid motor outcome")
        context=self.context_index(action) if context is None else context
        if not 0 <= context < len(TRAIN_OFFSETS):
            raise ValueError("Unknown training position")
        self.context_counts[action,context]+=1
        self.counts[action]+=1
        self.values[action]+=(reward-self.values[action])/self.counts[action]

    def save(self,path):
        path=Path(path)
        snapshot=dict(schema=2,slices=self.slices,profiles=PROFILES,counts=self.counts.tolist(),values=self.values.tolist(),
                      context_counts=self.context_counts.tolist(),training_offsets=TRAIN_OFFSETS,
                      rng_state=self.rng.bit_generator.state,
                      scope="Motor-profile reward averages, not neural weights")
        temporary=path.with_suffix(".partial")
        temporary.write_text(json.dumps(snapshot,indent=2))
        temporary.replace(path)
