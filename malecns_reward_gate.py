"""Test an explicit external reinforcement gate; preserve the failed original assay."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from malecns_conditioning import Assay
from malecns_probe import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 23, 101])
    parser.add_argument("--rounds", type=int, default=6, help="A+B presentations per training run")
    parser.add_argument("--train-only", action="store_true", help="Only train paired A; skips validation control runs")
    parser.add_argument("--resume", type=Path, help="Continue an existing checkpoint; requires one seed and --train-only")
    parser.add_argument("--output-dir", type=Path, help="Separate results directory")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.resume and (not args.train_only or len(args.seeds) != 1):
        parser.error("--resume requires --train-only and exactly one seed")
    start = time.perf_counter()
    assay = Assay(reward_gate=True)
    brain = assay.brain
    out = args.output_dir or (ROOT / "data/experiments/conditioning_reward_gate")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "intervention":"External reinforcement gate on plasticity, not neuron firing",
        "reason":"Original model produces endogenous PPL101 spikes to both cues, even without imposed reinforcement",
        "learning_rule":"Original eta=0.05, tau=1000ms, floor=0.1; only gate changed",
        "gate":"Accept PPL101 modulatory spikes only during imposed 80-140ms reinforcement; preserve 1.8ms delay",
        "scope":"Direct-KC conditioning; engineered plasticity, no learned behavior or biological validation",
        "neurons":brain.n, "anatomical_connections":brain.graph.nnz,
        "eligible_connections":len(assay.positions),
        "criterion":"Paired cue response falls by >=50%; other cue retains >=80%; both controls unchanged",
        "seeds":{}, "complete":False,
        "training_rounds":args.rounds, "cue_presentations_per_run":2*args.rounds,
        "controls_run":not args.train_only,
        "resume_checkpoint":str(args.resume.resolve()) if args.resume else None,
    }
    for seed in args.seeds:
        assay.set_cues(seed)
        brain.weights.data[assay.positions] = assay.original
        before = assay.evaluate()
        result = {"before":before, "arms":{}, "cue_body_ids":{
            k:brain.neurons.bodyId.iloc[v].tolist() for k,v in assay.cues.items()}}
        arms = ["paired_A"] if args.train_only else ["paired_A", "paired_B", "frozen", "no_reinforcement"]
        for arm in arms:
            print(f"Seed {seed}: {arm}", flush=True)
            brain.weights.data[assay.positions] = assay.original
            if args.resume:
                with np.load(args.resume,allow_pickle=False) as saved:
                    expected = {"pre_body":brain.neurons.bodyId.iloc[assay.pre],
                                "post_body":brain.neurons.bodyId.iloc[assay.post],
                                "cue_A":brain.neurons.bodyId.iloc[assay.cues["A"]],
                                "cue_B":brain.neurons.bodyId.iloc[assay.cues["B"]]}
                    for key,value in expected.items():
                        if not np.array_equal(saved[key],value):
                            raise ValueError(f"Checkpoint does not match this graph/cue seed: {key}")
                    weights = saved["trained"]
                    if (weights.shape != assay.original.shape or not np.isfinite(weights).all()
                        or not np.allclose(saved["original"],assay.original)
                        or np.any(weights < 0.1*assay.original-1e-7)
                        or np.any(weights > assay.original+1e-7)):
                        raise ValueError("Checkpoint has incompatible or invalid synaptic efficacies")
                    brain.weights.data[assay.positions] = weights
            starting_weights = brain.weights.data[assay.positions].copy()
            dan_spikes = 0
            for iteration in range(args.rounds):
                for cue in ["A", "B"]:
                    reinforced_cue = "B" if arm == "paired_B" else "A"
                    trial = assay.trial(cue,
                        reinforce=arm != "no_reinforcement" and cue == reinforced_cue,
                        plastic=arm != "frozen")
                    dan_spikes += trial["PPL101_spikes"]
                print(f"  Round {iteration+1}/{args.rounds}",flush=True)
            trained = brain.weights.data[assay.positions].copy()
            after = assay.evaluate()
            assert np.array_equal(trained, brain.weights.data[assay.positions])
            changed = int(np.count_nonzero(trained != assay.original))
            controls_ok = None
            if arm in ["frozen", "no_reinforcement"]:
                controls_ok = changed == 0 and before == after
                assert controls_ok, f"Control failed: {arm}"
            retention = {cue:(after[cue]["MBON11_spikes"]/before[cue]["MBON11_spikes"]
                              if before[cue]["MBON11_spikes"] else None)
                         for cue in ["A", "B"]}
            passed = controls_ok
            if arm in ["paired_A", "paired_B"]:
                target = arm[-1]
                other = "B" if target == "A" else "A"
                passed = (all(v is not None for v in retention.values())
                          and retention[target] <= 0.5 and retention[other] >= 0.8)
            result["arms"][arm] = {"after":after, "changed_connections":changed,
                "MBON11_response_retained_fraction":retention, "criterion_passed":bool(passed),
                "PPL101_training_spikes":dan_spikes}
            result["arms"][arm]["connections_changed_this_run"] = int(np.count_nonzero(trained != starting_weights))
            np.savez_compressed(out/f"seed{seed}_{arm}.npz", pre_body=brain.neurons.bodyId.iloc[assay.pre],
                post_body=brain.neurons.bodyId.iloc[assay.post], original=assay.original, trained=trained,
                cue_A=brain.neurons.bodyId.iloc[assay.cues["A"]],
                cue_B=brain.neurons.bodyId.iloc[assay.cues["B"]])
            print(f"  Response retained: {retention}; changed={changed}; pass={passed}", flush=True)
            report["seeds"][str(seed)] = result
            (out/"report.json").write_text(json.dumps(report, indent=2))
        result["passed"] = all(r["criterion_passed"] for r in result["arms"].values())
    report["complete"] = True
    report["all_seeds_passed"] = all(r["passed"] for r in report["seeds"].values())
    report["full_controlled_validation_passed"] = report["all_seeds_passed"] if not args.train_only else None
    report["wall_seconds"] = time.perf_counter()-start
    (out/"report.json").write_text(json.dumps(report, indent=2))
    print("Selected response checks passed:", report["all_seeds_passed"])
    if args.train_only:
        print("Control runs were skipped. This is not a new full validation.")
    print("Report:", out/"report.json")


if __name__ == "__main__":
    main()
