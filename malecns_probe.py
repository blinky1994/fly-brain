"""Full selected MaleCNS graph, approximate point-neuron response probe.

This is a numerical starting model, not a validated brain emulation or learning.
Direct Kenyon-cell stimulation bypasses sensory processing. Uncertain transmitter
and neuromodulator labels are explicitly omitted from fast transmission; their
anatomical edges remain in the saved graph. No synapse counts are modified.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy import sparse

ROOT = Path(__file__).resolve().parent


class Brain:
    def __init__(self):
        folder = ROOT / "data/processed"
        self.neurons = feather.read_table(folder / "neurons.feather").to_pandas()
        self.n = len(self.neurons)
        self.graph = sparse.load_npz(folder / "synapse_counts.npz").tocsc()
        assert self.graph.shape == (self.n, self.n)
        self.graph.sort_indices()
        assert self.neurons.bodyId.is_monotonic_increasing
        labels = self.neurons.consensus_nt.fillna("MISSING")
        # Chosen transmitter-sign approximation, not receptor-specific physiology.
        signs = labels.map({"acetylcholine": 1, "gaba": -1,
                           "glutamate": -1, "histamine": -1}).fillna(0)
        self.signs = signs.to_numpy(dtype=np.float32)
        self.omitted = labels[self.signs == 0].value_counts().to_dict()
        self.weights = self.graph.astype(np.float32)
        self.weights.data *= np.repeat(self.signs, np.diff(self.weights.indptr))
        self.weights.data *= 0.275
        self.dt = 0.1  # ms
        self.delay = round(1.8 / self.dt)
        self.refractory_steps = round(2.2 / self.dt)
        self.av = np.exp(-self.dt / 20)
        self.ag = np.exp(-self.dt / 5)
        self.reset()

    def reset(self):
        self.v = np.full(self.n, -52.0, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.queue = [np.empty(0, dtype=np.int32) for _ in range(self.delay)]
        self.clock = 0

    def step(self, drive):
        active = self.refractory == 0
        # Exact subthreshold solution for fixed drive and exponential synaptic state.
        v_next = (-52 + (self.v + 52)*self.av + drive*(1-self.av)
                  + self.g*(self.av-self.ag)/3)
        self.v[active] = v_next[active]
        self.g[active] *= self.ag
        slot = self.clock % self.delay
        arriving = self.queue[slot]
        if arriving.size:
            delta = np.asarray(self.weights[:, arriving].sum(axis=1)).ravel()
            self.g[active] += delta[active]
        fired = np.flatnonzero(active & (self.v > -45)).astype(np.int32)
        self.refractory[~active] -= 1
        self.v[fired] = -52
        self.g[fired] = 0
        self.refractory[fired] = self.refractory_steps
        self.queue[slot] = fired
        self.clock += 1
        if not np.isfinite(self.v).all() or not np.isfinite(self.g).all():
            raise RuntimeError("Non-finite neural state: reject this run.")
        return fired

    def trial(self, cue, duration_ms, stimulate):
        self.reset()
        counts = np.zeros(self.n, dtype=np.int64)
        drive = np.zeros(self.n, dtype=np.float32)
        activity = []
        for step in range(round(duration_ms/self.dt)):
            now = step*self.dt
            drive[cue] = 20 if stimulate and 20 <= now < duration_ms-20 else 0
            fired = self.step(drive)
            counts[fired] += 1
            activity.append(len(fired))
        mbon = self.neurons.type.fillna("").eq("MBON11").to_numpy()
        stimulated_counts = int(counts[cue].sum())
        return {
            "spikes_total": int(counts.sum()),
            "neurons_that_spiked": int(np.count_nonzero(counts)),
            "directly_stimulated_cell_spikes": stimulated_counts,
            "other_cell_spikes": int(counts.sum()) - stimulated_counts,
            "MBON11_spikes": {
                str(self.neurons.bodyId.iloc[i]): int(counts[i])
                for i in np.flatnonzero(mbon)
            },
            "largest_spiking_fraction_per_step": max(activity)/self.n,
        }, counts, np.asarray(activity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-ms", type=float, default=100)
    args = parser.parse_args()
    if not np.isfinite(args.duration_ms) or args.duration_ms < 60:
        parser.error("Use at least 60 ms.")
    start = time.perf_counter()
    print("Loading MaleCNS connection matrix...", flush=True)
    brain = Brain()
    learning = np.load(ROOT / "data/processed/learning_connections.npz")
    candidates = np.unique(learning["pre"])
    if not len(candidates):
        raise RuntimeError("No KC -> MBON11 source neurons found.")
    rng = np.random.default_rng(7)
    cue = np.sort(rng.choice(candidates, min(100, len(candidates)), replace=False))
    assert brain.neurons.type.iloc[cue].str.startswith("KC").all()
    output = ROOT / "data/experiments/neural_probe"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "experimental LIF response probe; no learning",
        "neurons": brain.n,
        "anatomical_connections": brain.graph.nnz,
        "dt_ms": brain.dt, "duration_ms": args.duration_ms,
        "membrane_tau_ms":20, "synaptic_tau_ms":5,
        "rest_and_reset_mV":-52, "threshold_mV":-45,
        "delay_ms":1.8, "refractory_ms":2.2,
        "synaptic_scale":0.275, "cue_drive":20,
        "fast_sign_assumption":"ACh positive; GABA/glutamate/histamine negative",
        "fast_transmission_omitted_neurons_by_label":brain.omitted,
        "cue_body_ids":brain.neurons.bodyId.iloc[cue].tolist(),
        "plasticity_enabled":False,
        "caveats":["Direct KC input bypasses senses", "No spontaneous drive or noise",
                   "No receptor-specific effects", "No neuromodulation yet",
                   "Parameters are model choices, not measured per-neuron values",
                   "A response is not evidence of learning or biological validity"],
    }
    for name, stimulate in [("no_input", False), ("direct_KC_input", True)]:
        print(f"Running {name}...", flush=True)
        stats, counts, trace = brain.trial(cue, args.duration_ms, stimulate)
        report[name] = stats
        np.savez_compressed(output/f"{name}.npz", counts=counts, spikes_per_step=trace)
        print(json.dumps(stats, indent=2), flush=True)
    report["wall_seconds"] = time.perf_counter()-start
    (output / "report.json").write_text(json.dumps(report, indent=2))
    assert report["no_input"]["spikes_total"] == 0, "Unexpected activity at rest"
    if report["direct_KC_input"]["other_cell_spikes"] == 0:
        print("No downstream spikes: investigate before adding learning.")
    print("Report:", output / "report.json")
    print("No weights were trained or changed on disk.")


if __name__ == "__main__":
    main()
