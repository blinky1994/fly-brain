# MaleCNS-driven knife decisions

The fly body was rebuilt because the earlier project was deleted. It runs in
`.venv-body` (Python 3.14, FlyGym 2.1.0). The full selected MaleCNS network remains
in `.venv` (Python 3.11). A local child process connects the two; neither environment
needs to be activated when using the commands below.

From Git Bash in this project:

```bash
.venv-body/Scripts/python.exe malecns_onion.py watch
```

The window alternates an aligned and an offset onion. Each trial runs a fresh
180-ms neural response using the existing trained synaptic efficacies, then runs
the resulting body action in live MuJoCo physics at half speed. While the brain
computes, the physical scene deliberately pauses and displays that status. This
is a discrete decision experiment, not continuous real-time brain/body control.
It does not play a recorded video. Close the window to stop both processes.

Keys apply on the next trial:

- **T**: trained A-pairing checkpoint.
- **U**: original untrained efficacies.
- **W**: reversed, B-pairing checkpoint.
- **A**: keep the onion aligned.
- **M**: keep it misaligned.
- **X**: alternate positions again.

To run the headless comparison:

```bash
.venv-body/Scripts/python.exe malecns_onion.py check
```

It saves `data/experiments/onion_bridge/report.json`. There are six positions per
condition: three within the encoded alignment tolerance and three outside it.
Aligned trials are correct only when an actual simulated cut event occurs;
misaligned trials are correct only when the neural decision is hold. The first
run produced 6/6 correct with the trained checkpoint, 3/6 untrained (always hold),
and 0/6 with reversed pairing. These are constructed interface checks, not an
estimate of general skill or biological validity.

## What the neural network does

The sensor encoder converts geometric lateral error <=0.12 mm into direct input
to cue A's 100 Kenyon cells; other positions activate cue B. The selected full
166,700-neuron graph runs with the approximate dynamics already documented in
`MALECNS_EXPERIMENTS.md`. A fixed decoder converts <=1 total MBON11 spike into
chop; more spikes mean hold. The threshold was chosen using the earlier
conditioning results, rather than independently validated. The decoder does not
inspect the alignment flag. Switching only the saved neural efficacies reverses
the decision, as the control demonstrates.

This transfers a previously trained cue response to the knife interface. It does
NOT train from successful cuts in the kitchen. Decisions run with plasticity and
reinforcement off. To repeat the preceding cue-training experiment, run:

```bash
.venv/Scripts/python.exe malecns_reward_gate.py
```

That replaces the gate experiment's reports/checkpoints. It trains all three cue
selections; the current knife interface uses seed 7.

## What is engineered

- The fly is rigidly supported; the knife is attached to its right front tarsus.
- Deterministic inverse kinematics defines the raise/lower motor primitive. Neural
  activity chooses whether to execute it, not the seven joint trajectories.
- The onion position is measured directly; there is no vision or smell model.
- Knife/board collision is simulated. Other fly collisions are disabled.
- Cutting means a downward blade-edge midpoint passage through the onion's target
  region; its appearance swaps to two pieces. There is no fracture, tissue
  deformation, resistance, or free-moving food.
- Both scene classes are intentionally easy to distinguish. Position variation
  still maps to the same two cues and does not demonstrate novel-cue generalization.

This is an assisted brain-to-body prototype, not an autonomously trained cooking
fly. The next substantial step would be online action-contingent reinforcement,
followed by more realistic sensing and manipulation.
