# Learn thin slicing before knife pickup

This is a new motor-training lesson. It chooses cutting settings based on the
physical result. The earlier `train_onion.py` still trains the separate binary
chop/hold decision; adding more episodes there cannot teach slicing.

## Watch actual training

From this project folder in Git Bash:

```bash
./.venv-body/Scripts/python.exe train_slicing.py --episodes 12 --speed 0.5
```

Start with the default **three-slice curriculum**. Each episode attempts three
consecutive slices on one fresh onion. The viewer shows three baseline onions,
then the requested training episodes, then three evaluation onions. Evaluation
does not update the policy. The viewer closes when the finite session finishes.

To keep learning while the viewer stays open:

```bash
./.venv-body/Scripts/python.exe train_slicing.py --episodes 12 --speed 1 --continuous
```

This repeats training blocks with a short evaluation between blocks. Closing the
viewer stops the run. Completed learning is saved after every training episode.
More attempts do not guarantee improvement once the available profiles are learned.

Use keys **1** for half speed, **2** for normal, **3** for 4x, and **0** to remove
deliberate playback delays. The viewer reports its phase, whether the action is
exploratory, the current stroke, clean-slice count, and the episode reward.
Physics and geometric planning can be slower than the requested playback speed.

For faster training without rendering:

```bash
./.venv-body/Scripts/python.exe train_slicing.py --headless --episodes 12
```

## Resume this motor policy

Slicing sessions are saved under `data/experiments/slicing_training/`. Their
checkpoint is **`latest.json`**, not the neural model's `latest.npz`.

```bash
checkpoint=$(find ./data/experiments/slicing_training -type f -name latest.json -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d ' ' -f 2-)
if [ -n "$checkpoint" ]; then
  ./.venv-body/Scripts/python.exe train_slicing.py --checkpoint "$checkpoint" --episodes 12 --speed 0.5
else
  echo "No slicing checkpoint yet. Start a fresh slicing session first."
fi
```

Resuming preserves outcome averages, attempt counts, and the exploration random
state. A new output folder is created; old saves remain intact. The console prints
that folder at startup. `config.json` records settings, `trials.jsonl` records each
attempt's measurements, and `report.json` compares baseline with final evaluation.
Use `--output-dir` only with a new empty directory inside this project.

The experimental `--slices 5` curriculum attempts a longer sequence. It is harder
and must be trained and evaluated separately; do not infer that success with
three means success with five. The checkpoint loader rejects a different slice
count or a different set of motor profiles. Use the default three first.

## What is being learned?

The motor learner tries six combinations of:

- A small correction to the knife's aim as it progresses across the onion.
- Time taken for a downward stroke.
- How far the holding pad presses into its target.

It initially tests each profile and records its actual reward. It then usually
chooses the profile with the highest average reward, with occasional exploration.
This is a small bandit learner, not a neural network that generates joint commands.
It does not modify the MaleCNS connectome or overwrite the older brain checkpoint.
The six profiles are supplied possibilities; the choice between them is learned
from fresh physical trials, not assigned by an expected-success label.

The geometric controller measures the remaining onion's position before each
stroke and solves intermediate leg poses along a straight path. It keeps the
knife edge approximately level, lifts the holding foot before reaching across,
and descends onto the onion. That controller is engineered. The learner adjusts
its settings; it does not discover inverse kinematics or visual perception.

## What counts as a clean slice?

The simulation distinguishes a **detached** piece from a **clean** slice. A clean
slice requires its joining seams to be released, a sufficiently aligned blade,
a full downward stroke, holding contact, and limited remainder slip.

The current checks require alignment error <=0.018 mm and plane-angle error <=8
degrees at fracture. Both ends of the knife edge must reach within 0.045 mm of
the board. Holding force must exceed 0.001 model units at fracture and at full
depth, with remainder slip below 0.08 mm at both times. The reward also penalizes
peak blade force above 25 model units and slip over the whole attempt. Blade–foot
contact above 0.001 model units, or an unreachable next stroke, makes the episode
fail. These numbers are chosen model thresholds, not calibrated food measurements.

Clean completion contributes most of the reward. Detachment alone earns only a
small partial score. The controller stops a failed attempt if the onion leaves
the reachable workspace; it does not teleport it back or count that as success.

## Onion realism and limitations

The scene uses a prepared half onion with a flat face on the board, curved red
skin detail, layered cut faces, a thinner knife, and separate physical slices.
Each piece has its own free joint, mass, friction, and collision geometry.
Blade contact and downward work release predefined connecting seams. There is
no replacement image or commanded separation impulse after a cut.

The target spacing is **0.05 mm in this miniature fly-scale model**, not a normal
kitchen onion's slice size. Thickness is preset by the mesh boundaries; the
learner learns to follow those boundaries, not to create arbitrary thicknesses.
The body is supported and the knife remains attached. The onion is rigid between
predefined seams: no tissue tearing, deformation, juices, or measured tissue
fracture mechanics are modeled. Appearance is more detailed, not photorealistic
or mechanically equivalent to real onion tissue.

The mechanics use MuJoCo's mesh/contact and equality-constraint facilities;
see the [official modeling documentation](https://mujoco.readthedocs.io/en/stable/modeling.html).

## Checks

Policy tests:

```bash
./.venv-body/Scripts/python.exe -m unittest test_slicing_policy
```

Physical controls, using a slicing checkpoint path you have selected:

```bash
./.venv-body/Scripts/python.exe check_slicing.py --checkpoint "$checkpoint"
```

The physical check expects three clean policy slices, no fracture without chopping,
no fracture when seams are locked, and no clean-slice credit without holding.
It saves before/after images and measurements in a new `slicing_checks` folder.
Use `--frozen` with `train_slicing.py` to compare against a fixed starting policy.
These are tests of this constructed task, not proof of a general cooking skill.
