# Left-foot holding and physical onion pieces

**Update:** outcome-based chop/hold training is now available with
`train_onion.py`. See [OUTCOME_TRAINING.md](OUTCOME_TRAINING.md) for the live
viewer, speed controls, and checkpoint continuation. The limitations below about
cue-only training describe the older `malecns_onion.py` mode.

Run the MaleCNS decision demo with the new body scene:

```bash
.venv-body/Scripts/python.exe malecns_onion.py watch --physical
```

The left front tarsus has a small frictional pad. It reaches the onion before the
right front leg lowers its attached knife, and maintains contact through the cut.
The torso remains supported. The viewer displays pad contact and seam state.
Existing keys still apply: T/U/W select trained/untrained/reversed neural weights;
A/M select aligned/misaligned onion positions; X resumes alternating positions.

## Physical model

The onion comprises two convex ellipsoid half meshes, each with its own free joint,
mass, and collision geometry. A weld connects the halves to each other before
fracture. Nothing welds the onion to the board or foot. The onion can slide and
roll; the holding pad acts through normal contact and friction.

The blade contacts the pieces and board. A custom damage proxy accumulates blade
normal contact force times downward edge travel. When it exceeds 0.02 in model
units, with current blade/onion contact and downward motion, the mutual weld is
disabled. Existing physical pieces then move independently; no replacement image,
teleport, or prescribed separation impulse is used. Mass is 0.00005 model mass
units per half; contact/weld solref is 0.0005 seconds with damping ratio 1.
These parameters were selected to make a stable millimetre-scale demonstration,
not fitted to measurements of onion tissue.

This is a rigid, predefined-fracture approximation. There are no deformable
layers, arbitrary cuts, tissue fracture mechanics, or realistic cutting-force
calibration. Only the added holding pad represents left-foot contact; the rest of
the detailed fly meshes do not provide a complete collision-safety model.

## Checks and their limits

```bash
.venv-body/Scripts/python.exe onion_physics.py check
.venv-body/Scripts/python.exe malecns_onion.py check --physical
```

Physical checks require left-pad contact before fracture, positive holding force
at fracture, no pad/blade contact, no fracture from holding alone, and no fracture
when the seam is locked. Held/unheld and sideways-disturbance comparisons are
recorded too. The centered onion also cuts without holding, so these tests do NOT
establish that holding is necessary. The current hold is not a robust general
grasp; larger disturbances can dislodge the onion.

The bridge's aligned trials now require both a cut and holding contact before/at
fracture to count as correct. Small +/-0.08 mm offsets use an engineered adjustment
of the left-foot target. Larger offsets disable the reach rather than attempting
an unsupported grasp. The neural input still encodes only the initial geometric
alignment class, not live force or contact feedback. The brain retains its
previously conditioned chop/hold decision; both joint trajectories remain fixed.

Results are separate from the old visual-only demo:

- `data/experiments/physical_onion/report.json`: physical checks.
- `data/experiments/physical_onion/disturbances.json`: additional nudge comparisons.
- `data/experiments/physical_onion_bridge/report.json`: neural/body interface checks.

Watching does not train. Online reinforcement from cut success, slip, holding
contact, and excessive forces is still a separate next task. This update supplies
those measurable physical signals without claiming the holding skill was learned.

## Watching the current cue training live

```bash
.venv-body/Scripts/python.exe malecns_onion.py watch --physical --train-live --rounds 6
```

This new mode begins with untrained connections unless `--checkpoint PATH` is
specified. It displays two baseline body trials, runs one A+B cue-training round,
then displays the next two body trials with updated weights. The cycle repeats
for the requested number of rounds. Afterwards the viewer continues evaluation
without further updates and labels training complete.

The window remains responsive while neural computation runs in a separate
process. Physics deliberately pauses during neural decisions and training.
The overlay reports completed rounds and connections changed in the last round.
Mode-switching keys T/U/W are disabled during a live training session; A/M/X still
choose the displayed onion positions. These display choices do not alter the
training curriculum: every training round pairs cue A with reinforcement and
presents cue B without reinforcement.

Completed rounds are saved atomically to `latest.npz` inside a new timestamped
folder under `data/experiments/live_training`. Neural updates and body results
are logged separately as `rounds.jsonl` and `body_trials.jsonl`. Closing during a
round keeps the last fully saved checkpoint. To continue that checkpoint, add
`--checkpoint path/to/latest.npz` on a later launch; a new session/output folder
is created and the displayed round count starts at zero for the new session.

Crucially, this trains the existing artificial cue association in the live viewer.
It is not online reinforcement from successful cuts, and it does not learn the
holding or knife trajectories. Physical trials show the effect of the changing
neural response through the existing fixed decoder and motor primitive.
