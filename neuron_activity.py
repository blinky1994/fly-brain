"""Read-only per-neuron telemetry and measured atlas; no simulation or RNG calls."""
from pathlib import Path
import numpy as np


def spike_frame(body_ids, counts, time_ms, window_ms=20):
    ids=np.asarray(body_ids)
    counts=np.asarray(counts)
    if (ids.shape!=counts.shape or ids.ndim!=1 or not np.isfinite(counts).all()
            or (counts<0).any() or (counts!=np.floor(counts)).any()):
        raise ValueError('Invalid per-neuron spike counts')
    active=np.flatnonzero(counts)
    return dict(neuron_spikes=[[int(ids[i]),int(counts[i])] for i in active],
                neural_time_ms=float(time_ms),window_ms=float(window_ms))


def write_atlas(neurons,motor_indices,sensor_indices,path):
    xyz=np.full((len(neurons),3),np.nan,dtype=np.float64)
    for i,point in enumerate(neurons.somaLocation):
        if point is not None:
            value=np.asarray(point,dtype=float)
            if value.shape==(3,) and np.isfinite(value).all(): xyz[i]=value
    motor=np.zeros(len(neurons),dtype=bool); motor[motor_indices]=True
    sensor=np.zeros(len(neurons),dtype=bool); sensor[sensor_indices]=True
    path=Path(path)
    with path.open('wb') as stream:
        np.savez_compressed(stream,schema=1,body_ids=neurons.bodyId.to_numpy(dtype=np.int64),
            xyz=xyz,types=neurons.type.fillna('untyped').to_numpy(dtype=str),
            classes=neurons.superclass.fillna('unclassified').to_numpy(dtype=str),
            sides=neurons.somaSide.fillna(neurons.rootSide).fillna('?').to_numpy(dtype=str),
            motor=motor,sensor=sensor,
            source='Local MaleCNS v1.0 annotations: somaLocation; missing coordinates omitted')
    return dict(path=str(path),positioned=int(np.isfinite(xyz).all(axis=1).sum()),total=len(neurons))


class ActivityState:
    """Replace every frame; never accumulate old spikes or invent activity."""
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as a:
            if int(a['schema'])!=1: raise ValueError('Unknown neuron atlas format')
            for key in ['body_ids','xyz','types','classes','sides','motor','sensor']:
                setattr(self,key,a[key].copy())
        if (self.body_ids.ndim!=1 or self.xyz.shape!=(len(self.body_ids),3)
                or not np.all(np.diff(self.body_ids)>0)):
            raise ValueError('Invalid atlas ordering or positions')
        self.positioned=np.isfinite(self.xyz).all(axis=1)
        self.counts=np.zeros(len(self.body_ids),dtype=np.int32)
        self.time_ms=0.
        self.window_ms=20.
        self.has_frame=False

    def clear(self):
        self.counts[:]=0
        self.has_frame=False

    def update(self,frame):
        pairs=np.asarray(frame['neuron_spikes'])
        if pairs.size==0: pairs=np.empty((0,2),dtype=np.int64)
        if pairs.ndim!=2 or pairs.shape[1]!=2 or not np.isfinite(pairs).all():
            raise ValueError('Invalid activity frame')
        ids=pairs[:,0].astype(np.int64)
        values=pairs[:,1]
        if (not np.array_equal(ids,pairs[:,0]) or len(np.unique(ids))!=len(ids)
                or (values<=0).any() or (values!=np.floor(values)).any() or (values>10000).any()):
            raise ValueError('Duplicate IDs or invalid counts')
        indices=np.searchsorted(self.body_ids,ids)
        if (indices>=len(self.body_ids)).any() or not np.array_equal(self.body_ids[indices],ids):
            raise ValueError('Activity refers to neurons outside this atlas')
        time_ms=float(frame['neural_time_ms']); window=float(frame['window_ms'])
        if not np.isfinite([time_ms,window]).all() or time_ms<0 or window<=0:
            raise ValueError('Invalid neural timing')
        self.counts[:]=0
        self.counts[indices]=values.astype(np.int32)
        self.time_ms,self.window_ms=time_ms,window
        self.has_frame=True
