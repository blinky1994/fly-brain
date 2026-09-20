# Train movement with the MaleCNS connectome

## Current first lesson: reach the green target

`train_connectome.py` now defaults to `--lesson reach`. The knife starts attached
to the right front leg. A green sphere sits 0.10 mm toward the onion from its
starting tip. The objective is to finish within 0.025 mm of its centre and remain
there for at least 0.20 seconds. Cutting is disabled during this lesson.

The score rewards smaller average and final distances and time held inside the
target, with effort and unsafe contact penalties. Training still changes existing
MaleCNS motor-input connection strengths; the neural spikes produce the actions.
There is no supplied reaching trajectory. Target coordinates affect the reward
only: this is learning a fixed reach, **not visual tracking of arbitrary targets**.

Every run measures its starting checkpoint, trains, then tests with learning off
from three starting postures (right front-leg first axis: 0 and ±0.006 radians).
The goal stays fixed. `reaching_demonstrated` is true only when all three finish
with the required hold. An accepted update or a moving leg is not proof of reaching.
This lesson does not automatically advance to slicing.

```bash
# Train the reaching lesson without windows.
./.venv-body/Scripts/python.exe train_connectome.py --lesson reach --headless --episodes 100 --seconds 2 --output-dir data/experiments/reach_practice_1

# Continue that checkpoint with body and neuron windows; close the body to stop.
./.venv-body/Scripts/python.exe train_connectome.py --lesson reach --neurons --continuous --episodes 12 --seconds 2 --checkpoint data/experiments/reach_practice_1/latest.npz

# Test saved weights without changing them.
./.venv-body/Scripts/python.exe train_connectome.py --lesson reach --headless --frozen --episodes 1 --seconds 2 --checkpoint data/experiments/reach_practice_1/latest.npz
```

Use a fresh output-directory name each run. Inspect `report.json` for the three
evaluations and `trials.jsonl` for distances, dwell time, and `reach_success`.
Old direct-motor checkpoints can warm-start this lesson, but their old reward is
discarded and measured again. Use the same lesson and duration for comparisons.
For the earlier combined reaching/holding/cutting reward, explicitly select
`--lesson slice`; descriptions of cutting scores below refer to that mode.

This is the current experiment. The older `train_slicing.py` chose among
hand-supplied movements and does not run MaleCNS. Its 9/9 slicing result does
not carry over to this experiment.

## Start or resume without a viewer

Run these commands from the prepared project folder in Git Bash:

```bash
# Start fresh. The neural worker is started automatically.
./.venv-body/Scripts/python.exe train_connectome.py --headless --episodes 100 --seconds 2
```

To resume the most recently saved run (including the validation runs):

```bash
checkpoint=$(find ./data/experiments -type f -name latest.npz -path '*connectome*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d ' ' -f 2-)
if [ -n "$checkpoint" ]; then
  ./.venv-body/Scripts/python.exe train_connectome.py --headless --episodes 100 --seconds 2 --checkpoint "$checkpoint"
else
  echo "No connectome movement checkpoint found; use the fresh command above."
fi
```

Every session gets its own directory. Completed trials save `latest.npz`
atomically, including accepted neural efficacies and random state. The original
MaleCNS anatomical data is unchanged. Old chop/hold NPZ and profile JSON saves
are incompatible with this new motor interface. Back up `data` separately:
experiment data is excluded from Git.

One baseline checks the starting weights, then the requested training attempts
run, followed by three frozen evaluations. `--seconds 2` is physical simulation
time per attempt, not wall-clock time. Headless training removes rendering and
playback delays; neural simulation still takes time.

For a viewer later, omit `--headless`. Add `--continuous` to repeat training
blocks until the window closes. Do not start a second long run unnecessarily;
it competes for CPU and memory. Keys 1/2/3/0 change playback speed; drag to orbit,
scroll to zoom, and close the window to stop after keeping completed learning.
Add `--neurons` to open the connected neuron-activity window alongside the body.
Closing only the neuron window leaves body training running. See
[NEURON_VIEWER.md](NEURON_VIEWER.md) for its separate controls. Headless runs can
use `--record-neurons` to save per-neuron activity without opening any window.

## How the connectome controls movement

The local model runs all **166,700 selected neurons and 25,582,938 anatomical
connections** using the existing approximate spiking-neuron dynamics. Each
command uses freshly computed activity; responses are never cached.

1. Joint angles stimulate annotated front-leg proprioceptors. Holding contact
   supplies a tactile signal. This is an engineered sensor encoder, not camera
   vision or a measured model of each sensory neuron's tuning.
2. Activity propagates through the existing connectome.
3. **89 annotated front-leg motor neurons**, grouped by named muscles and side,
   produce opposing signals for the 14 front-leg joint commands. The muscle-to-
   FlyBody-axis/sign conventions are provisional; the connectome does not supply
   calibrated muscle mechanics.
4. Joint commands move the simulated fly. There are no stored chopping poses,
   chosen movement profiles, or inverse-kinematics stroke planner in this loop.

The neural simulation advances 20 milliseconds for each 0.1-second physical
control interval. These are separate clocks, chosen for this experiment, not a
validated biological time correspondence. Fast neural state persists between
commands and resets between attempts.

## What training changes

Training perturbs continuous efficacy gains on **22,611 existing signed
connections entering the selected motor neurons**, grouped into 26 motor
populations. It runs a physical attempt and keeps the candidate only if its
score improves over the incumbent on the same starting scene. Signs and
anatomical connectivity remain unchanged. The fixed body decoder is not trained.

This is reward-guided search of internal connectome parameters, not an additional
movement-selection learner. It is also not a biological dopamine/STDP claim:
the connectome is a wiring dataset and does not specify a learning rule.
Most synapses remain fixed in this first motor experiment.

The score currently rewards getting the knife and holding foot closer to their
goals, with larger rewards for actual completed cuts and clean thin slices.
It penalizes excessive movement, slipping, and knife–holding-pad contact.
Small score improvements without cuts mean better early movement, not slicing.
Changing the episode duration remeasures the baseline before learning resumes.

## The onion is initially one object

`intact_onion.py` creates one rounded, slightly flat-bottomed rigid body. It has
no hidden slices, internal seam constraints, or visible cut faces initially.
A contact-aligned incision temporarily uses an explicit cutting-resistance
approximation instead of rigid blade contact with that body. After a complete
stroke crosses the local cross-section, meshes are created at the measured
knife plane. Pieces inherit the parent's pose and velocity, conserve total
mass, and receive no separation impulse. Layered faces appear only after a cut.

The current splitter supports approximately parallel cuts along the onion's
local y direction, at continuously measured positions. It is not arbitrary
3D fracture, deformable tissue, or a calibrated food simulation. The supported
fly body and attached knife are still simplifications. Knife pickup is later.

This uses MuJoCo's [model editing and state-preserving recompilation](https://mujoco.readthedocs.io/en/latest/programming/modeledit.html).

## How to read results

- `trials.jsonl`: physical scores, actual cuts, neural spike counts, proposed
  connection changes, and whether an update was retained.
- `activity.jsonl`: fresh motor-population activity and the resulting joint
  commands for each attempt.
- `neural_model.json`: selected biological neuron IDs and the explicit body mapping.
- `latest.npz`: the most recently saved accepted neural parameters.
- `report.json`: baseline and frozen evaluation after the requested block ends.
- `neural_worker.log`: errors from the neural process, if any.

`candidate_connections_changed` describes exploration. Only an accepted candidate
changes the saved incumbent. `retained_connections_changed` is zero during frozen
evaluation and rejected attempts. The full graph still runs in either case.

## Verified so far

The first 12 physical learning attempts accepted three internal neural updates.
The centered two-second score increased from approximately 0.00974 to 0.00986.
**No slices were learned or produced in these neural trials.** Motor activity is
still sparse, and robust reaching, holding, and slicing remain unsolved.

Separate checks verify that fresh motor spikes cause nonzero commands, silencing
the motor readout gives zero commands, frozen evaluation does not change weights,
and checkpoint restoration reproduces the neural state parameters.

The onion remains one body at rest. A **scripted test fixture**, deliberately
separate from training, completed one contact-driven cut at a measured thickness
of about 0.060 mm. This validates the cutting mechanism; it does not demonstrate
learned cutting. Distances are miniature fly-model millimetres.

```bash
./.venv/Scripts/python.exe -m unittest test_connectome_motor
./.venv-body/Scripts/python.exe -m unittest test_intact_onion
./.venv-body/Scripts/python.exe check_intact_cut.py
```
