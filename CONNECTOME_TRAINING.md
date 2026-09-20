# Bench-press training with MaleCNS

## Current grip setup

The earlier contact-only setup let the legs slip off. Passive point constraints
now secure both feet to bar anchors, with no scripted lift or bar actuator. The
viewer labels this ASSISTED GRIP: grasping is provided, not learned. Original
checkpoints can warm-start this changed task, but scores must be remeasured.
New logs record `assisted_grip`, `grip_errors_mm`, and each grip force.

## Initial validation (older contact-only scene)

The first completed run with the fly on its back (`bench_validation_2`) used
eight training attempts. Frozen evaluations reached peak lifts of about
0.0237–0.0238 mm, returned to the bottom, and passed **0/3** lift-and-hold tests.
All evaluations retained zero weight changes. The 0.15 mm goal has not been
learned. `bench_validation_1` was an earlier upright-body development check;
do not use it as a bench-press checkpoint or comparison.
The same accepted checkpoint with motor output disabled
(`bench_silenced_validation_1`) produced zero lift in every trial. All eleven
physics, neuron-viewer, and neural-controller tests passed at this transition.

Use Git Bash from the prepared project folder. The neuron viewer remains part
of this experiment; see [NEURON_VIEWER.md](NEURON_VIEWER.md).

## Train without windows

```bash
./.venv-body/Scripts/python.exe train_connectome.py --headless --episodes 100 --seconds 2
```

Every fresh run starts from the anatomical weights with sensor gain 4, an
experimental calibration that makes both front-leg outputs responsive. It is
not a measured biological current. `--sensory-gain` explicitly overrides this
setting (0.1–4); otherwise a resumed checkpoint restores its saved calibration.
The full graph remains present. Learning adjusts 26 gains on 22,611 existing
signed incoming motor-neuron connections, with no added anatomical connections.

## Resume in the two live windows

Replace the path below with the checkpoint from your bench session:

```bash
checkpoint="data/experiments/bench_validation_2/latest.npz"
./.venv-body/Scripts/python.exe train_connectome.py --neurons --continuous --episodes 12 --seconds 2 --checkpoint "$checkpoint"
```

The first trial measures the incumbent, then twelve candidates are tested,
then three trials evaluate the accepted weights without learning. Continuous
mode repeats those blocks until the body window closes. `learning: false`
during a baseline or evaluation is expected. Completed trials save atomically;
an interrupted unfinished trial is not saved.

`--seconds 2` is body simulation time. Each 0.1 seconds of body time uses a
fresh 20 ms of neural simulation. These clocks intentionally differ.
`--speed` only changes playback pacing; headless mode removes rendering cost.

## Check whether it learned

```bash
checkpoint="data/experiments/bench_validation_2/latest.npz"
# Learning disabled: does the accepted controller perform a lift?
./.venv-body/Scripts/python.exe train_connectome.py --headless --frozen --episodes 1 --seconds 2 --checkpoint "$checkpoint"

# Causal control: neurons run, but their motor commands are forced to zero.
./.venv-body/Scripts/python.exe train_connectome.py --headless --frozen --silence-motor-output --episodes 1 --seconds 2 --checkpoint "$checkpoint"
```

The bar has a passive slide joint and gravity, with no actuator or upward spring.
Two passive foot-to-bar constraints transmit the front-leg forces. Guides enforce its position and levelness;
the supported torso and non-front-leg posture are not learned.

Reward = 60% final normalised height + 40% average normalised height + up to 1
for the final hold, minus a small accumulated effort penalty. Heights are capped
at the 0.15 mm goal for scoring. Success requires the final 0.20 seconds above
that goal with both grip constraints active, both anchor errors below 0.02 mm, and
absolute bar speed below 0.2 mm/s. Grip forces are averaged over each 0.02-second
body step. This is an assisted attachment, not a muscle-force or grasping model.
The sensory input uses a shared mean grip-force signal plus joint angles; no target or camera pixels enter
the neural controller. Lowering, repeated reps, gripping and balance are later lessons.

Read `lifting_demonstrated` in `report.json`: it requires all three frozen tests
to succeed, including ±0.006-radian starting-posture perturbations. Compare the
baseline and final evaluations, not only the best training score. Check
`retained_connections_changed` is zero on evaluations. The disabled-output
control should not lift the bar. Even success from an untrained controller
would not, on its own, demonstrate learning.

## Checks

```bash
./.venv-body/Scripts/python.exe -m unittest test_bench_press test_neuron_activity
./.venv/Scripts/python.exe -m unittest test_connectome_motor
```

The mechanical contact test uses a Jacobian only as a test fixture to verify
that secured legs can lift the passive bar. The trainer never calls that fixture
or supplies a lifting trajectory. Neural tests check fresh activity, motor-output
ablation, frozen weights, save/resume and read-only viewer telemetry.
