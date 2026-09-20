# Watch the neurons driving onion training

The neuron window reads the **same MaleCNS worker that commands the fly**.
It does not start FLYBOARD's separate simulator, generate synthetic activity,
or change the training rule. The renderer is adapted from
[FLYBOARD by NullLabTests](https://github.com/NullLabTests/flybrain).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the retained MIT license.

## Start from the most recent movement checkpoint

From the prepared project folder in Git Bash:

```bash
checkpoint=$(find ./data/experiments/connectome_motor -type f -name latest.npz -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d ' ' -f 2-)
if [ -n "$checkpoint" ]; then
  ./.venv-body/Scripts/python.exe train_connectome.py --neurons --continuous --episodes 12 --seconds 2 --checkpoint "$checkpoint"
else
  echo "No checkpoint yet. Use the fresh command below."
fi
```

Fresh training, with both windows:

```bash
./.venv-body/Scripts/python.exe train_connectome.py --neurons --continuous --episodes 12 --seconds 2
```

Both windows use the existing `.venv-body` packages. The neural worker runs in
`.venv`. No new anatomy download is needed: positions come from the local
MaleCNS `somaLocation` annotation. Keep one training session running at a time
to avoid competing for resources or following different checkpoints.

## Neuron-window controls

| Control | Effect |
| --- | --- |
| Drag with the left mouse button | Rotate the anatomy |
| Mouse wheel | Zoom |
| Click a point | Inspect its body ID, cell type, side, role, and spike count |
| 1 | Whole positioned CNS |
| 2 | Nerve-cord-related populations |
| 3 | The connected sensory and motor populations |
| R | Reset the camera |
| Q, Escape, or window close | Close only the neuron display; training continues |

Closing the **body window** stops the training process and closes both windows.
Completed learning is saved after each full attempt. The existing body-window
speed keys apply only to body playback; 1/2/3 mean filters in the neuron window.

## What the dots mean

- Orange dots fired during the most recently received **20 ms neural window**.
  Brightness uses a fixed spike-count scale capped at five spikes per window.
  These are windowed spike counts, not an animation of exact spike timestamps.
- Dim dots show anatomy only. They do not imply firing.
- The atlas contains **139,662 measured soma positions** for the 166,700 simulated
  neurons. All **89 motor readout neurons** have positions. The other **27,038**
  cells remain simulated but are not placed in the 3D cloud. The header reports
  firing cells without positions explicitly. No coordinates are invented.
- The display shows cell bodies, not neuron branches or individual synaptic edges.
  Coordinates undergo one rigid rotation and a uniform scale, preserving shape.
- Header times distinguish the neural clock from body time. An episode reset
  clears activity. If no fresh frame arrives for three wall-clock seconds, glow
  clears and the window says it is waiting. Closing the worker closes the display.

Movement learning is still experimental and has not demonstrated learned slicing.
The orange activity is from our spiking simulation, not recordings of a living fly.

## Record without opening windows

```bash
./.venv-body/Scripts/python.exe train_connectome.py --headless --record-neurons --episodes 12 --seconds 2
```

With either `--neurons` or `--record-neurons`, the session contains:

- `neuron_atlas.npz`: biological IDs, measured coordinates, labels, and input/output roles.
- `neuron_atlas.json`: coordinate coverage and the atlas path.
- `activity.jsonl`: each command's sparse `[bodyId, spikeCount]` entries, neural
  time, window length, group activity, and the resulting joint commands.

Every new frame replaces the preceding activity; omitted cells have zero spikes
in that window. Older runs only recorded group totals, so individual-neuron
activity cannot be reconstructed from those logs.

The added telemetry has a regression test showing identical motor commands and
unchanged neural weights with recording enabled or disabled. Rendering can slow
wall-clock throughput, so use headless recording for faster training.

```bash
./.venv/Scripts/python.exe -m unittest test_connectome_motor
./.venv-body/Scripts/python.exe -m unittest test_neuron_activity
```
