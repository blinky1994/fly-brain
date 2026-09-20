# Fly bench press

A simulated fly learns to lift a guided barbell using the MaleCNS connectome.
The body and live neuron viewer use the same neural simulation.

This replaces the onion experiment. The first goal is one **lift and hold**,
not a full bench-press repetition. The fly is supported on its back. Orange
grip pads on its front legs are secured to the passive bar by assisted grip
constraints; vertical guides keep
it level. The green line marks the target height. There is no scripted lift,
bar actuator, learned grasp, or movement library. The viewer explicitly labels
the passive attachment as ASSISTED GRIP.
The orange pads and grip anchors sit at the distal claw tips (`tarsus5`),
not the proximal foot joints. The bar starts aligned to those measured mesh tips.
The bar displays three large "20 KG" plates per side. These are gym-style visual
labels; the experimental fly-scale training load is unchanged (not 120 real kg).

## Open the demo (Git Bash)

Run from this prepared project folder:

```bash
./.venv-body/Scripts/python.exe train_connectome.py --neurons --continuous --episodes 12 --seconds 2
```

That starts fresh learning. To resume the newest normal bench session:

```bash
checkpoint=$(find ./data/experiments/bench_press -type f -name latest.npz -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d ' ' -f 2-)
if [ -n "$checkpoint" ]; then
  ./.venv-body/Scripts/python.exe train_connectome.py --neurons --continuous --episodes 12 --seconds 2 --checkpoint "$checkpoint"
else
  echo "No saved bench session yet; use the fresh command above."
fi
```

Close the **body window** to stop training. Completed attempts are saved.
Closing just the neuron window leaves training running.

Body controls: drag to orbit, wheel to zoom; 1 = slower, 2 = normal,
3 = faster, 0 = no playback delay. Neural computation still takes real time.
Neuron controls: 1 = whole CNS, 2 = nerve cord, 3 = sensory/motor cells;
click a dot to inspect it. [Neuron viewer guide](NEURON_VIEWER.md).

## What is it learning?

1. Joint positions and a simplified grip-force signal stimulate sensory neurons.
2. The full selected MaleCNS graph simulates spikes.
3. Annotated front-leg motor-neuron groups command fourteen joint velocities.
4. Secured legs can lift the bar against gravity.
5. After each attempt, the trainer scores bar height and holding time.
6. It keeps changes to existing neural connection strengths only if they improve
   the score. It restores the previous accepted settings otherwise.

Success requires lifting **0.15 mm** and holding it for the final **0.20 seconds**
with both assisted grips within 0.02 mm of their anchors and low bar speed. The score rewards final
and average height plus the final hold, with a small effort penalty. Guides
handle balance, so the fly does not yet learn to keep the bar level itself.
These are simulator-scale distances and an experimental load, not a claim
about real fly strength. A higher score or glowing neurons is not proof of success.

The model uses measured wiring, with engineered sensory currents, simplified
neuron dynamics, provisional muscle-to-joint mapping, and a search-based learning
rule. It has no visual understanding of the gym or goal.

## Where learning is saved

Each normal run creates `data/experiments/bench_press/<session>/`:

| File | What it contains |
| --- | --- |
| `latest.npz` | Accepted connection strengths, sensor gain, counters, random state |
| `trials.jsonl` | Rewards, lift heights, contacts, successes and weight changes |
| `report.json` | Frozen evaluation results after each training block |
| `activity.jsonl` | Neural commands and spikes; per-neuron frames with the viewer |
| `config.json` / `neural_model.json` | Session settings and neural mapping |

Back up `data/` separately: it is excluded from Git. The original connectome
files under `data/processed/` are not changed by training. Cooking checkpoints
are preserved but rejected by this task because the body interface changed.

## Project map

- `bench_press.py`: body, passive bar, contacts, reward and success criteria.
- `train_connectome.py`: training loop, saves, evaluation and both viewers.
- `malecns_motor_brain.py`: sensory encoding, motor readout and learned strengths.
- `malecns_probe.py`: full-graph spiking dynamics.
- `build_connectome.py`: prepares the local anatomical graph from raw data.
- `motor_viewer.py`: body display.
- `neuron_activity.py` / `neuron_viewer.py`: neuron telemetry and display.

[Training guide](CONNECTOME_TRAINING.md) explains frozen tests and controls.
[Third-party credits](THIRD_PARTY_NOTICES.md) include the FLYBOARD renderer.

## Cleanup and existing setup

The obsolete cooking scripts, tests and guides were removed from the active
project. A recovery copy, including the old guides, is stored locally as
`data/legacy_cooking_source_20260921.zip`. Existing training data and downloaded
anatomy/body assets were kept. That archive is excluded from Git too.

Commands assume the existing Windows `.venv-body` (MuJoCo/FlyGym/viewers),
`.venv` (neural worker), local MaleCNS data and FlyBody mesh assets. Cloning the
source alone does not install those dependencies or download the datasets.
