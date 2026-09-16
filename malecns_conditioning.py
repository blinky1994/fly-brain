"""Experimental direct-KC conditioning on the selected MaleCNS graph.

Plasticity is an imposed phenomenological rule, not supplied by the connectome.
No sensory perception, natural motor behavior, or organism-level learning claim.
"""
import json
import time
from pathlib import Path

import numpy as np

from malecns_probe import Brain, ROOT


class Assay:
    def __init__(self, seed=7, reward_gate=False):
        self.reward_gate = reward_gate
        self.brain = Brain()
        b = self.brain
        with np.load(ROOT / "data/processed/learning_connections.npz") as edges:
            self.pre = edges["pre"].astype(int)
            self.post = edges["post"].astype(int)
        self.positions = []
        for pre, post in zip(self.pre, self.post):
            start, stop = b.weights.indptr[pre:pre+2]
            offset = np.searchsorted(b.weights.indices[start:stop], post)
            assert offset < stop-start and b.weights.indices[start+offset] == post
            self.positions.append(start+offset)
        self.positions = np.array(self.positions)
        assert len(np.unique(self.positions)) == len(self.positions)
        self.original = b.weights.data[self.positions].copy()
        assert np.all(self.original > 0), "Plastic KC outputs must be excitatory here"
        types = b.neurons.type.fillna("")
        self.dan = np.flatnonzero(types.eq("PPL101"))
        self.mbon = np.flatnonzero(types.eq("MBON11"))
        assert len(self.dan) == len(self.mbon) == 2
        self.mod_masks = {
            int(d): (b.neurons.somaSide.iloc[self.post].to_numpy()
                     == b.neurons.somaSide.iloc[d])
            for d in self.dan
        }
        self.set_cues(seed)

    def set_cues(self, seed):
        rng = np.random.default_rng(seed)
        sources = rng.permutation(np.unique(self.pre))
        assert len(sources) >= 200
        self.cues = {"A": np.sort(sources[:100]), "B": np.sort(sources[100:200])}

    def trial(self, cue, reinforce=False, plastic=False, capture_eligibility=False):
        b = self.brain
        b.reset()
        eligibility = np.zeros(b.n, dtype=np.float32)
        drive = np.zeros(b.n, dtype=np.float32)
        spikes = np.zeros(b.n, dtype=np.int64)
        eligibility_decay = np.exp(-b.dt/1000)
        dopamine_queue = [np.empty(0, dtype=int) for _ in range(b.delay)]
        for step in range(round(180/b.dt)):
            ms = step*b.dt
            drive[self.cues[cue]] = 20 if 20 <= ms < 120 else 0
            drive[self.dan] = 20 if reinforce and 80 <= ms < 140 else 0
            fired = b.step(drive)
            spikes[fired] += 1
            eligibility *= eligibility_decay
            eligibility[fired] += 1
            slot = step % b.delay
            if plastic:
                for dan in dopamine_queue[slot]:
                    mask = self.mod_masks[int(dan)]
                    positions = self.positions[mask]
                    old = b.weights.data[positions]
                    b.weights.data[positions] = np.maximum(
                        0.1*self.original[mask],
                        old*np.exp(-0.05*eligibility[self.pre[mask]])
                    )
            # Optional engineering intervention: only dopamine spikes during an
            # externally supplied reinforcement window can cause plasticity.
            # Neural activity is unchanged; this is NOT inferred fly physiology.
            gate = not self.reward_gate or (reinforce and 80 <= ms < 140)
            dopamine_queue[slot] = (np.intersect1d(fired, self.dan) if gate
                                    else np.empty(0, dtype=int))
        if capture_eligibility:
            self.last_eligibility = eligibility[self.pre].copy()
        return {"MBON11_spikes":int(spikes[self.mbon].sum()),
                "MBON11_by_body":{str(b.neurons.bodyId.iloc[i]):int(spikes[i]) for i in self.mbon},
                "PPL101_spikes":int(spikes[self.dan].sum()),
                "total_spikes":int(spikes.sum())}

    def evaluate(self):
        return {cue:self.trial(cue) for cue in self.cues}


def main():
    started = time.perf_counter()
    assay = Assay()
    b = assay.brain
    out = ROOT / "data/experiments/conditioning"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "scope":"direct-KC circuit assay; not sensory learning or behavior",
        "neurons":b.n, "anatomical_connections":b.graph.nnz,
        "eligible_existing_connections":len(assay.positions),
        "cue_body_ids":{k:b.neurons.bodyId.iloc[v].tolist() for k,v in assay.cues.items()},
        "model":"Same LIF/sign assumptions as malecns_probe.py",
        "plasticity":{"rule":"dopamine-gated multiplicative depression",
                      "eta":0.05, "eligibility_tau_ms":1000,
                      "weight_floor_fraction":0.1, "delay_ms":1.8,
                      "target_assignment":"PPL101 to same-side MBON11 inputs; engineered assumption",
                      "training_pairs":6, "trial_ms":180,
                      "cue_window_ms":[20,120], "reinforcement_window_ms":[80,140],
                      "reset":"Fast state and traces cleared between trials; efficacies retained"},
        "fast_transmission_omitted_labels":b.omitted,
        "arms":{},
    }
    for arm in ["paired_A", "frozen", "paired_B_control"]:
        print(f"Running {arm}...", flush=True)
        b.weights.data[assay.positions] = assay.original
        before = assay.evaluate()
        train_stats = []
        for _ in range(6):
            for cue in ["A", "B"]:
                reward_cue = "B" if arm == "paired_B_control" else "A"
                train_stats.append(assay.trial(cue, reinforce=(cue == reward_cue),
                                               plastic=(arm != "frozen")))
        trained_weights = b.weights.data[assay.positions].copy()
        after = assay.evaluate()  # Both reinforcement and plasticity OFF.
        assert np.array_equal(trained_weights, b.weights.data[assay.positions])
        changed = int(np.count_nonzero(trained_weights != assay.original))
        if arm == "frozen":
            assert changed == 0 and before == after, "Frozen control changed"
        np.savez_compressed(out/f"{arm}_synapses.npz", pre_body=b.neurons.bodyId.iloc[assay.pre],
                            post_body=b.neurons.bodyId.iloc[assay.post],
                            original=assay.original, trained=trained_weights)
        result = {"before":before, "after":after, "changed_connections":changed,
                  "training_PPL101_spikes":sum(x["PPL101_spikes"] for x in train_stats)}
        report["arms"][arm] = result
        print(json.dumps(result, indent=2), flush=True)
        (out/"report.json").write_text(json.dumps(report, indent=2))
    def suppression(arm, cue):
        r = report["arms"][arm]
        return r["before"][cue]["MBON11_spikes"]-r["after"][cue]["MBON11_spikes"]
    selective = (suppression("paired_A", "A") > max(0, suppression("paired_A", "B"))
                 and suppression("paired_B_control", "B") > max(0, suppression("paired_B_control", "A")))
    report["cue_selective_suppression_in_this_assay"] = bool(selective)
    report["biological_learning_demonstrated"] = False
    report["wall_seconds"] = time.perf_counter()-started
    (out/"report.json").write_text(json.dumps(report, indent=2))
    print("Cue-selective suppression in this single deterministic assay:", selective)
    print("This is not yet a behavioral-learning result or a validated biological model.")
    print("Report:", out/"report.json")


if __name__ == "__main__":
    main()
