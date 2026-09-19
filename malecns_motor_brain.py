"""Full MaleCNS simulation -> annotated front-leg motor outputs.

Annotated front-leg proprioceptors receive engineered joint-angle encoding.
Annotated muscle motor-neuron groups drive joint velocities through provisional
axis/sign conventions. Reward searches EXISTING incoming motor-neuron efficacies.
No movement library, MBON-to-joint projection, or independent learned controller.
This is not validated physiology; anatomy alone supplies no dynamics or learning rule.
"""
import json
from pathlib import Path
import sys
import numpy as np
from malecns_probe import Brain, ROOT


class MotorBrain:
    def __init__(self, motor_names, observation_size, checkpoint=None, seed=41):
        self.brain = Brain()
        b = self.brain
        self.motor_names = list(motor_names)
        self.observation_size = int(observation_size)
        if self.observation_size != 22:
            raise ValueError('Expected the version-1 physical sensor interface')
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        n = b.neurons
        types = n.type.fillna('').str.strip()
        front = n.superclass.eq('vnc_motor') & n.somaNeuromere.eq('T1') & n.subclass.eq('fl')
        # Muscle labels ground the grouping; the FlyBody axis/sign mapping remains
        # provisional and explicitly recorded. This is not a muscle force model.
        pairs = [
            ('Sternal anterior rotator MN','Sternal posterior rotator MN'),
            ('Sternal adductor MN','Pleural remotor/abductor MN'),
            ('Tergopleural/Pleural promotor MN','Pleural remotor/abductor MN'),
            ('Fe reductor MN','Tergotr. MN'),
            ('Tr flexor MN','Tr extensor MN'),
            ('Ti flexor MN','Ti extensor MN'),
            ('Ta depressor MN','Ta levator MN'),
        ]
        if len(motor_names)!=14:
            raise ValueError('Expected seven front-leg axes on each side')
        self.output_groups, self.group_labels, self.readout = [], [], []
        for side, prefix in [('L','lf_'),('R','rf_')]:
            for axis, pair in enumerate(pairs):
                i=len(self.readout)
                if prefix not in motor_names[i]:
                    raise ValueError('Unexpected motor ordering')
                indices=[]
                for label in pair:
                    key=side+':'+label
                    if key not in self.group_labels:
                        group=np.flatnonzero((front & n.somaSide.eq(side) & types.eq(label)).to_numpy())
                        if not len(group): raise ValueError('Missing annotated motor group '+key)
                        self.group_labels.append(key)
                        self.output_groups.append(group)
                    indices.append(self.group_labels.index(key))
                self.readout.append(indices)
        self.readout=np.array(self.readout)
        self.motor_neurons=np.unique(np.concatenate(self.output_groups))
        output_map=np.full(b.n,-1,dtype=int)
        for i,g in enumerate(self.output_groups): output_map[g]=i
        self.sensor_groups=[]
        proprio=n['class'].eq('mechanosensory_proprioceptive') & n.entryNerve.fillna('').str.contains('ProLN')
        self.tactile=[]
        for side in ['L','R']:
            sources=np.flatnonzero((proprio & n.rootSide.eq(side)).to_numpy())
            if len(sources)<14: raise ValueError('Not enough annotated front-leg proprioceptors')
            self.sensor_groups.extend(np.array_split(sources,14))
            self.tactile.append(np.flatnonzero((n['class'].eq('mechanosensory_tactile') &
                n.entryNerve.fillna('').str.contains('ProLN') & n.rootSide.eq(side)).to_numpy()))
        # Retain the full graph. Search only signed efficacies already entering
        # the annotated motor cells; never add an anatomical connection.
        target=output_map[b.weights.indices]
        self.positions=np.flatnonzero((target>=0) & (b.weights.data!=0))
        self.pre=np.searchsorted(b.weights.indptr,self.positions,side='right')-1
        self.post=b.weights.indices[self.positions]
        self.edge_group=target[self.positions]
        self.original=b.weights.data[self.positions].copy()
        channels=len(self.output_groups)
        self.gains=np.zeros(channels)
        self.rounds = self.accepted = 0
        self.best_reward = None
        self.pending = None
        if checkpoint:
            with np.load(checkpoint,allow_pickle=False) as saved:
                if str(saved['schema'])!='malecns-direct-motor-v1':
                    raise ValueError('Use a direct-motor checkpoint, not a chop/hold or profile save')
                if saved['motor_names'].tolist()!=self.motor_names or int(saved['observation_size'])!=observation_size:
                    raise ValueError('Motor/sensor interface mismatch')
                if int(saved['seed'])!=seed or not np.array_equal(saved['pre_body'],b.neurons.bodyId.iloc[self.pre]):
                    raise ValueError('Neural encoding or anatomical input mismatch')
                if not np.array_equal(saved['post_body'],b.neurons.bodyId.iloc[self.post]):
                    raise ValueError('Anatomical output mismatch')
                if not np.array_equal(saved['original'],self.original):
                    raise ValueError('Underlying model changed')
                if saved['motor_groups'].tolist()!=self.group_labels:
                    raise ValueError('Motor-neuron annotation mapping changed')
                self.gains = saved['gains'].copy()
                if self.gains.shape!=(channels,) or not np.isfinite(self.gains).all() or np.max(abs(self.gains))>1.1:
                    raise ValueError('Invalid learned gains')
                expected=(self.original*np.exp(self.gains[self.edge_group])).astype(self.original.dtype)
                if not np.array_equal(saved['trained'],expected):
                    raise ValueError('Checkpoint efficacy/gain mismatch')
                self.rounds, self.accepted = int(saved['rounds']),int(saved['accepted'])
                self.rng.bit_generator.state = json.loads(str(saved['rng_state']))
        self._apply(self.gains)

    def _apply(self,gains):
        self.brain.weights.data[self.positions] = self.original*np.exp(gains[self.edge_group])

    def begin(self,learn=False,silence=False):
        if self.pending is not None:
            raise ValueError('Finish the preceding trial first')
        candidate = (np.clip(self.gains+self.rng.normal(0,.24,len(self.gains)),-1.1,1.1)
                     if learn else self.gains.copy())
        self.pending = (candidate,bool(learn))
        self._apply(candidate)
        self.brain.reset()
        self.rates = np.zeros(len(self.gains))
        self.total_spikes = self.output_spikes = self.windows = 0
        self.silence = bool(silence)
        return dict(candidate=bool(learn),cached_response=False)

    def act(self,observation):
        if self.pending is None:
            raise ValueError('Begin a neural trial first')
        obs = np.asarray(observation,dtype=float)
        if obs.shape!=(self.observation_size,) or not np.isfinite(obs).all():
            raise ValueError('Invalid physical observation')
        drive = np.zeros(self.brain.n,dtype=np.float32)
        # The six target-error observations are used ONLY by the external reward,
        # never injected into proprioceptors as fictitious visual perception.
        angles=obs[6:20]
        encoded=np.r_[angles[:7],-angles[:7],angles[7:],-angles[7:]]
        for group,value in zip(self.sensor_groups,encoded):
            drive[group]=16+12*float(np.clip(value,-1,1))
        for group in self.tactile:
            drive[group]=30*float(np.clip(obs[-1],0,1))
        counts = np.zeros(self.brain.n,dtype=np.int32)
        # Fast state persists across commands within an episode. Fresh simulation,
        # including all selected neurons/connections, for every 20-ms window.
        for _ in range(round(20/self.brain.dt)):
            fired = self.brain.step(drive)
            counts[fired] += 1
        group_activity = np.array([counts[g].mean() for g in self.output_groups])
        self.rates = .5*self.rates+.5*group_activity
        self.total_spikes += int(counts.sum())
        self.output_spikes += int(counts[self.motor_neurons].sum())
        self.windows += 1
        # Fixed opponent readout; no stored poses, stroke timing, or IK targets.
        action = np.tanh(self.rates[self.readout[:,0]]-self.rates[self.readout[:,1]])
        if self.silence:
            action[:] = 0
        return dict(action=action.tolist(),window_ms=20,spikes=int(counts.sum()),
                    motor_spikes=int(counts[self.motor_neurons].sum()),
                    motor_activity=group_activity.tolist(),cached_response=False)

    def finish(self,reward,reference=False):
        if self.pending is None or not np.isfinite(reward):
            raise ValueError('Invalid/unmatched physical outcome')
        candidate,learn = self.pending
        before = self.brain.weights.data[self.positions].copy()
        previous = (self.original*np.exp(self.gains[self.edge_group])).astype(self.original.dtype)
        accepted = bool(learn and self.best_reward is not None and reward>self.best_reward+1e-9)
        if learn:
            if self.best_reward is None:
                raise ValueError('Measure the incumbent on this scene before learning')
            self.rounds += 1
        if accepted:
            self.gains = candidate
            self.best_reward = float(reward)
            self.accepted += 1
        elif reference and not learn and not self.silence:
            self.best_reward = float(reward)
        self._apply(self.gains)
        changed = int(np.count_nonzero(self.brain.weights.data[self.positions]!=previous))
        self.pending = None
        return dict(reward=float(reward),accepted=accepted,learning=learn,
                    candidate_connections_changed=int(np.count_nonzero(before!=previous)),
                    retained_connections_changed=changed,training_rounds=self.rounds,
                    accepted_rounds=self.accepted,total_spikes=self.total_spikes,
                    motor_spikes=self.output_spikes,neural_windows=self.windows,
                    neural_simulated_ms=self.windows*20,best_reward=self.best_reward)

    def save(self,path):
        if self.pending is not None:
            raise ValueError('Only completed trials can be saved')
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError('Save motor learning inside this project')
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix('.partial')
        b = self.brain
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,schema='malecns-direct-motor-v1',motor_groups=self.group_labels,gains=self.gains,
                pre_body=b.neurons.bodyId.iloc[self.pre],post_body=b.neurons.bodyId.iloc[self.post],
                original=self.original,trained=b.weights.data[self.positions],seed=self.seed,
                motor_names=self.motor_names,observation_size=self.observation_size,
                rounds=self.rounds,accepted=self.accepted,rng_state=json.dumps(self.rng.bit_generator.state))
        temporary.replace(path)
        return dict(checkpoint=str(path))


def main():
    brain = None
    for line in sys.stdin:
        try:
            req = json.loads(line)
            op = req.pop('operation')
            if op=='init':
                if brain is not None:
                    raise ValueError('Already initialized')
                brain = MotorBrain(**req)
                result = dict(ready=True,neurons=brain.brain.n,connections=brain.brain.graph.nnz,
                              plastic_connections=len(brain.positions),outputs=len(brain.motor_neurons),
                              motor_groups=brain.group_labels,motor_body_ids=brain.brain.neurons.bodyId.iloc[brain.motor_neurons].tolist(),
                              joint_readout=[dict(joint=name,positive=brain.group_labels[p],negative=brain.group_labels[n])
                                             for name,(p,n) in zip(brain.motor_names,brain.readout)],
                              proprioceptor_body_ids=[brain.brain.neurons.bodyId.iloc[g].tolist() for g in brain.sensor_groups],
                              learning='reward search of existing motor-input efficacies; annotated motor readout')
            elif brain is None or op not in ['begin','act','finish','save']:
                raise ValueError('Unknown operation or uninitialized brain')
            else:
                result = getattr(brain,op)(**req)
        except Exception as exc:
            result = dict(error=f'{type(exc).__name__}: {exc}')
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
