# Training from simulated cutting outcomes

Run from this project folder in Git Bash on Windows:

```bash
./.venv-body/Scripts/python.exe train_onion.py --episodes 24 --speed 4
```

This opens the live MuJoCo viewer, shows six baseline trials, trains for 24
physical episodes, and shows six evaluation trials without exploration or
updates. Afterwards it keeps showing evaluation until you close the window.
The display identifies exploratory actions, training/evaluation phases, rewards,
and changed connections. Physics pauses during neural computation; the viewer
stays responsive. Close it to stop; the last completed training update is saved.

For maximum throughput while watching, use `--speed 0`. For maximum throughput
without a window:

```bash
./.venv-body/Scripts/python.exe train_onion.py --headless --episodes 24
```

`--speed 4` removes the old half-speed pacing and targets 4x simulated time for
body playback, limited by CPU speed. It does not make neural computation 4x
faster. Zero removes deliberate playback waits. Headless also removes rendering.
All modes keep the original neural and physical timesteps and full selected
neural graph. A deterministic neural response and its eligibility trace are
reused only for the same cue with unchanged weights; any weight change clears
the cache. `--no-cache` disables this optimization for comparisons. Physics and
reward are always computed afresh; cached decisions are identified in the logs.

Each run creates a new folder under `data/experiments/outcome_training`, containing
`latest.npz`, settings, detailed physical outcomes, trial logs, and the completed
baseline/evaluation report. `--output-dir` can name a new empty folder inside the
project. To continue learning, pass `--checkpoint` followed by the saved NPZ path.
Weights resume; episode counts and the reproducible exploration sequence start
anew in the new session. Old runs are preserved. Without a checkpoint, weights
start untrained, even if previous sessions exist.

## Learning rule and scope

The existing alignment encoder selects one of two artificial neural cues. MBON11
spikes select chop or hold through the existing fixed decoder. With probability
0.35, training tries the opposite action to discover physical outcomes. The
reward function does not receive the alignment class or a desired action.

A chop earns +1 only when the physical seam breaks with left-pad contact before
and during fracture, prefracture slip <=0.12 mm, blade force <=50 model units,
and no blade-pad force above 0.001 model units. A miss or failure of those criteria
earns -1. Holding earns 0 (blade-pad contact still earns -1). These thresholds are
engineering choices for this scene, not calibrated tissue or safety limits.

The decision trial retains presynaptic eligibility on existing KC-to-MBON11
connections. After a chop, the physical reward drives an engineered signed update:
`w *= exp(-0.15 * reward * eligibility)`, clipped to 10%-200% of original weight.
Successful chops lower excitation into the inhibitory chop/hold decoder; failed
chops increase it. Holding does not update weights. This is a new outcome-gated
rule, distinct from the older imposed dopamine/cue-pairing assay. It is not an
inferred biological learning rule. The checkpoint layout stays compatible with
the earlier viewer.

This learns when to execute the existing movement. It does not learn holding,
knife trajectories, vision, or a general cutting skill. A failed movement can
teach avoidance but cannot repair that movement. Training alternates 0 and
0.6 mm offsets; evaluation includes +/-0.08 and +/-0.5 mm variations that still
map to the same two cues. Evaluation scores are constructed interface checks,
not broad generalization evidence.

Use `--frozen` for a no-learning control with the same exploration sequence.
Use `--exit-after-training` to close the viewer after the final evaluation.
The old `malecns_onion.py --train-live` remains the separate cue-pairing demo.

## Verified local run

The 16-episode run in `data/experiments/outcome_validation_2` improved from 3/6
baseline trials to 6/6 evaluation trials. The matched frozen run in
`data/experiments/outcome_frozen_validation` stayed at 3/6, and its saved weights
were exactly unchanged. Training included both rewarded held cuts and a penalized
misaligned chop. Training plus baseline/evaluation took about 119 seconds after
loading on this machine; this is not a guaranteed runtime.

`outcome_cache_validation/report.json` records exact equality of fresh and cached
spike counts and eligibility traces for an untrained cue. The fresh calculations
took about 2.3-2.6 seconds each; the cache lookup took under a millisecond. Overall
speedup is smaller because physical trials still run and updates invalidate the
cache. Unit tests cover reward failures, signed updates, frozen behavior, cache
invalidation, and rejecting duplicate or mismatched outcomes.
