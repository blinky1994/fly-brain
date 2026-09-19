# Fly onion training — beginner's guide

Repository: **fly-chop-onion**. The repository contains the source code, tests,
and guides. Large downloaded data, body assets, training runs/checkpoints, and
the local Python environments are excluded from Git. The commands below assume
the existing prepared Windows project folder; cloning this repository alone
does not recreate that environment. Back up `data` separately to preserve your
training checkpoints.

This project runs a simulated fly, knife, and onion. You can watch the fly try
actions, receive a score from the physical result, and change its future decisions.

**There are now two training lessons:** the original chop/hold decision and a
new thin-slicing motor lesson. Neither learns from camera images or discovers
joint movements from scratch.

## New: train clean thin slices

From this project folder in **Git Bash**:

```bash
./.venv-body/Scripts/python.exe train_slicing.py --episodes 21 --speed 0.5
```

This lesson starts with **three consecutive 0.05 mm slices in the miniature
simulation**. It learns which aim correction, cutting speed, and holding pressure
produce better physical results. The onion has separate physical pieces and
visible layers. A detached piece only counts as clean after a complete, aligned,
held cut with limited slip. This uses an engineered quality proxy, not measured
onion tissue behavior.

Add `--continuous` to train in blocks while the viewer stays open. Each block
includes evaluation with learning off, then training resumes. Closing the viewer
stops the session and keeps the latest completed update. Without `--continuous`,
the window closes after the final evaluation. Use `--headless` for faster runs.

**Slicing saves `latest.json` under `data/experiments/slicing_training`.** These
are motor-policy checkpoints, separate from the old neural `latest.npz` files.
The new lesson chooses among supplied movement profiles; it does not update
MaleCNS synapses. Slice thickness comes from preset fracture boundaries.

See [SLICING_TRAINING.md](SLICING_TRAINING.md) for resuming, speed keys, physical
checks, and limitations. The experimental five-slice curriculum is a harder,
separate task. The rest of this README explains the original chop/hold lesson.

## 1. Open the project

Open **Git Bash** (or a Git Bash tab in **Windows Terminal**). Copy this line,
paste it, and press Enter:

```bash
cd /c/Users/user/Desktop/Workspace/fly
```

All the commands below assume you have done that. Copy only the lines inside
the code boxes, not the surrounding headings or backticks.

The project already contains its Python environments and data. You do not need
to activate an environment or install packages to use this existing setup.
Close an existing training viewer before starting another session so the two
runs do not compete for your computer's resources.

## 2. Continue your most recent training

**Continuing requires a checkpoint.** A checkpoint is a saved copy of the learned
connection strengths. Starting without one resets learning to the untrained
state, even when old sessions exist on disk.

After opening the project as above, paste this whole block into Git Bash:

```bash
checkpoint=$(find ./data/experiments/outcome_training -type f -name latest.npz -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d ' ' -f 2-)
if [ -n "$checkpoint" ]; then
  printf 'Continuing from: %s\n' "$checkpoint"
  ./.venv-body/Scripts/python.exe train_onion.py --checkpoint "$checkpoint" --episodes 24 --speed 4
else
  echo "No saved outcome-training checkpoint found. Use the fresh-start command below first."
fi
```

This finds the most recently saved checkpoint in the normal outcome-training
folder, opens the viewer, and performs **24 additional training episodes**.
It saves the new learning in a new folder, preserving the old session.

The episode counter starts over for each session. The learned weights carry
over, but this is not a frame-by-frame continuation of the old simulation:
the scene resets and the reproducible exploration sequence restarts.

If you used a custom output folder, or want a particular saved run, pass that
checkpoint explicitly. For example, this resumes the verified 16-episode run
included in this folder:

```bash
./.venv-body/Scripts/python.exe train_onion.py --checkpoint "./data/experiments/outcome_validation_2/latest.npz" --episodes 24 --speed 4
```

The automatic selection chooses the **newest**, not necessarily the best,
checkpoint. Check a session's `report.json` when comparing results.

## 3. Start fresh instead

To watch the fly learn from untrained connections:

```bash
./.venv-body/Scripts/python.exe train_onion.py --episodes 24 --speed 4
```

This preserves your old files. It simply does not load their learned weights.

### Watch actual training slowly

To see the learning attempts clearly, start a fresh session at half-speed body
playback. From the project folder in **Git Bash**, run:

```bash
./.venv-body/Scripts/python.exe train_onion.py --episodes 24 --speed 0.5 --exit-after-training
```

Close older viewer windows first so you do not confuse their completed
demonstrations with the new training session.

The new window shows **six baseline trials first**, then **24 actual training
episodes**, then **six evaluation trials**. It closes automatically afterwards.
Baseline is not training, so seeing the fly hold still at the beginning is normal.
The half-speed setting slows body playback; it does not change the learning rule
or simulation timestep.

Look for these signs while watching:

- **Phase: `training`** in the viewer, and **`learning: true`** in the terminal:
  learning is enabled for this attempt.
- **Exploration**: the fly is trying the opposite of its current decision.
- **Reward**: the score from the physical attempt, such as `+1` for a held cut.
- **Connections updated** in the viewer, or **`connections_changed`** in the log:
  a positive number confirms that connection strengths actually changed.

Not every training attempt changes weights. Holding does not update them, and
some weights may already have reached their limits. Learning changes the
chop/hold decision; the leg movements remain programmed.

If you see **`complete - evaluation only`**, **`learning: false`**, and an episode
counter continuing above 24 in another run, it is repeating demonstrations,
not continuing to train. Those terminal records are normal logs, not errors.
The command above avoids that endless demonstration phase.

This command deliberately starts **untrained**, making the change in behavior
easier to see. It preserves previous saves. To continue existing learning instead,
add `--checkpoint` with a saved checkpoint path as described in section 2; a model
that already solves the task may show little or no further change.

## 4. What you will see

A session has four phases:

| Viewer phase | What happens | Does learning change? |
| --- | --- | --- |
| `baseline` | Six trials measure the starting behavior. | No |
| `training` | The requested number of episodes, with exploration and physical rewards. | Yes, unless `--frozen` was used |
| `evaluation` | Six trials test the resulting decisions without exploration. | No |
| `complete - evaluation only` | The viewer repeats demonstrations until you close it. | No |

**An episode** means one reset, one decision, one physical attempt, and its result.
`--episodes 24` means 24 training attempts, plus six baseline and six evaluation
trials. It does not mean 24 successful cuts.

The fly may hold still at first. It needs to try a chop before a successful cut
can teach it anything. During training, **exploration** sometimes tries the
opposite of its current decision. A trained fly can therefore still make a
deliberately exploratory mistake. Evaluation switches exploration off.

The scene pauses at **Computing neural decision** while the brain calculation
runs. This is expected; the viewer should remain responsive. The display also
shows the reward and how many connections changed.

Close the viewer to stop. The last completed training update is saved
automatically; an unfinished attempt may not be included. There is no Save button
to press. A final report is written only after the scheduled evaluation finishes.
For a run without a viewer, press **Ctrl+C** in its terminal to interrupt it;
the last completed checkpoint remains on disk.

## 5. Make it faster, or slow it down to watch

| Option | Meaning |
| --- | --- |
| `--speed 0.5` | Target half-speed body playback to make training attempts easier to follow. |
| `--speed 1` | Target normal simulated-time playback for the body. |
| `--speed 4` | Target four-times-speed body playback; the default. |
| `--speed 0` | Keep the viewer, but remove deliberate playback delays. Zero means maximum speed here. |
| `--headless` | Run without a viewer or playback delays. Best for throughput. |
| `--episodes 60` | Run more training attempts. This does not speed up each attempt. |
| `--exit-after-training` | Close the viewer after the final evaluation. |

For maximum speed while watching a **fresh** run:

```bash
./.venv-body/Scripts/python.exe train_onion.py --episodes 24 --speed 0
```

For faster **continued** training without a window, use the same checkpoint
selection block from section 2, but replace the `train_onion.py` command inside
the `if` block with:

```bash
./.venv-body/Scripts/python.exe train_onion.py --checkpoint "$checkpoint" --episodes 60 --headless
```

The `$checkpoint` variable exists only in the Bash session where you ran the
selection block. In a new terminal, use the complete section 2 block with the
training command replaced as shown above. Keep the `if` check so an empty
checkpoint path never starts a run.

The speed setting controls waiting between body movements. It cannot make an
expensive brain or physics calculation instantaneous. Actual playback speed is
limited by your CPU. These options keep the same simulation timesteps and neural
network; they do not trade accuracy for a larger timestep.

The program also **caches** neural responses: if the cue and learned weights are
unchanged, it can reuse the same deterministic answer. When weights change, it
discards that cache. It still simulates the body and computes the reward afresh.
`--no-cache` disables this optimization for testing; normally leave it enabled.

More episodes do not guarantee more improvement. This is a small, two-choice
task, so learning can reach its limits quickly. A zero connection-change count
can mean that the fly held still, learning is off, or weights reached their limit.

## 6. How it works, step by step

Think of the project as three connected parts: a brain that chooses an action,
a simulated body that performs it, and a scoring rule that judges the result.

1. **Measure the onion position.** The program reads its geometric offset from
   the knife path. It does not recognize an onion from an image.
2. **Turn that position into a cue.** Positions within 0.12 mm activate cue A;
   other positions activate cue B. Each cue stimulates a selected group of
   simulated neurons.
3. **Run the brain.** The selected MaleCNS graph contains about 166,700 neurons
   and 25.6 million anatomical connections. The code uses simplified electrical
   dynamics to calculate which neurons fire. These dynamics are modeling choices,
   not a complete reproduction of a living fly's brain.
4. **Read a decision.** A programmed rule looks at the output neurons called
   MBON11. At most one spike means chop; more spikes mean hold. A **spike** is
   a simulated neuron firing. The rule translating spikes into actions is fixed.
5. **Explore sometimes.** During training, the program tries the opposite action
   with probability 0.35, or roughly 35%. This allows a fly that initially always
   holds to discover a successful chop.
6. **Run the body.** MuJoCo, the physics simulator, calculates movement, contact,
   friction, and the onion's breakable seam. The left holding motion and right
   knife motion are programmed movement sequences.
7. **Score the physical result.** The score comes from what happened in the
   simulation, rather than from being told which cue should mean chop.
8. **Adjust connections.** A successful chop makes that cue more likely to
   produce a chop later. A failed chop pushes the decision toward holding.
   The next episode uses the updated connection strengths.

### The reward

| Result | Reward |
| --- | --- |
| Onion seam breaks with holding contact and within the configured slip/force limits | `+1` |
| Chop misses, fails to break the seam, cuts without holding, or exceeds those limits | `-1` |
| Hold without blade–foot contact | `0` |
| Blade–foot contact above the configured threshold | `-1` |

These are engineering rules for this particular scene, not measured biological
rewards or real-world safety limits. Holding does not update connections in the
current rule, including when a contact penalty is recorded.

### What exactly gets saved and learned?

A **weight** is the strength of a connection between simulated neurons. The
project changes eligible existing connections from Kenyon cells (the cue-input
neurons here) to MBON11 output neurons. It does not retrain every connection in
the full graph.

An **eligibility trace** is a short-lived record of which input connections were
active during the decision. It lets the later reward affect those connections.
A successful chop reduces their weights, lowering MBON11 output and favoring
chop under the existing decoder. A failed chop increases them, favoring hold.
Weights have upper and lower limits. This signed learning rule is deliberately
engineered; it is not a claim about how a real fly learns to use tools.

The checkpoint stores the learned weights and the neuron/cue identifiers needed
to load them again. It is a saved learning state, not a video or a whole frozen
copy of the running simulation.

## 7. Find your saves and understand the result

At startup, the terminal prints `Session:` followed by the new session's folder.
Normal runs are saved here:

```text
data/experiments/outcome_training/<long session number>/
```

The long number is just a unique session name. Open that folder in File Explorer:

| File | What it is for |
| --- | --- |
| `latest.npz` | Resume training from this file. It is not meant to be read in Notepad. |
| `config.json` | Settings used for this session. |
| `trials.jsonl` | One record per trial, with phase, decision, reward, and physical measurements. |
| `outcomes.jsonl` | Detailed decision and reward records from the brain worker. |
| `report.json` | Starting and final evaluation scores, written when evaluation finishes. |

You can open JSON and JSONL files in a text editor. JSONL means one JSON record
per line. In the report, compare `baseline_correct` with `evaluation_correct`.
Both scores are out of six: three aligned cases should produce held cuts and
three offset cases should produce hold decisions.

The verified 16-episode run improved from **3/6 to 6/6**. A matched run with
learning disabled stayed at **3/6**, and its weights were unchanged. This supports
learning in this constructed task. Six easy test cases do not establish a general
cooking skill, and the test positions still map to the same two cues.

To only watch a saved model, without training it, this command uses the verified
checkpoint. Replace its checkpoint path to watch another run:

```bash
./.venv-body/Scripts/python.exe malecns_onion.py watch --physical --checkpoint "./data/experiments/outcome_validation_2/latest.npz"
```

That is a separate demonstration viewer. Its older keyboard controls do not
apply to `train_onion.py`.

## 8. Common questions

**“Why does it keep running after training?”**  
The default viewer continues demonstrating the learned decisions. When the phase
says `complete - evaluation only`, it is no longer training. Close it, or use
`--exit-after-training` next time.

**“Why doesn't it get better after hundreds of episodes?”**  
It only chooses between two fixed actions for two encoded cue classes. It cannot
invent a better grip, repair the knife motion, or learn a new sensing system.
Those require extending the controller and the task.

**“Does hold mean the left foot stops moving?”**  
No. Hold means do not execute the chopping motion. In the physical scene, the
programmed left-foot reach can still run for an aligned onion.

**“Why is the resumed baseline already good?”**  
Baseline means before this session's new learning. When resuming, it evaluates
the weights you loaded, so it can start at 6/6.

**“The command says it can't find Python or a file.”**  
Check that you ran the `cd` line in section 1 and copied the command exactly.
Keep the quotes around paths. The two `.venv` folders and `data` are required;
copying only the Python scripts to a new computer is not enough.

**“It says the output folder must be empty.”**  
The program protects previous results. Omit `--output-dir` to get a new session
folder automatically. Use `--checkpoint` to read an old save; do not reuse that
old folder as the output destination.

**“I see an error or the viewer closes unexpectedly.”**  
Keep the terminal error text and the printed session path. For brain-worker
errors, the message may also point to a log under `data/experiments/onion_bridge`.
An empty brain log by itself is normal. Share the error and session path when
asking for help; do not delete your checkpoints to troubleshoot.

## 9. Which files do what?

| File or folder | Role |
| --- | --- |
| `train_slicing.py` | Live or headless thin-slice motor training, including continuous mode. |
| `thin_onion.py` | Layered physical slices, straight-path control, and cut-quality measurements. |
| `slicing_policy.py` | Learns which motor profile gets the best physical reward; saves JSON checkpoints. |
| `check_slicing.py` | Physical controls and before/after renders for a slicing checkpoint. |
| `train_onion.py` | Main entry point for outcome training and its viewer. |
| `onion_outcome.py` | Reward rules and the weight-update calculation. |
| `malecns_onion_brain.py` | Neural worker, decisions, caching, and checkpoints. |
| `malecns_conditioning.py` | Neural trials and activity/eligibility calculations. |
| `malecns_probe.py` | Loads the neural graph and runs simplified neuron dynamics. |
| `onion_physics.py` | Holding contact, onion halves, fracture approximation, physical measurements. |
| `onion_scene.py` | Base fly, knife, board, and programmed movements. |
| `malecns_onion.py` | Older viewer and the connection to the separate neural process. |
| `.venv-body` | Python environment for the body and viewer. |
| `.venv` | Python environment for the neural model. The body launches it automatically. |
| `data/processed` | Prepared neural graph and neuron data. |
| `data/experiments` | Saved runs, checkpoints, and reports. |

There are two Python environments because the body and brain use different
dependency setups. You normally launch only the body-side command shown above;
it starts and communicates with the brain process for you.

For technical details, see [OUTCOME_TRAINING.md](OUTCOME_TRAINING.md). For the
physical model's limitations, see [PHYSICAL_ONION.md](PHYSICAL_ONION.md).
[ONION_BRIDGE.md](ONION_BRIDGE.md) and
[MALECNS_EXPERIMENTS.md](MALECNS_EXPERIMENTS.md) describe the earlier experiments.
The older `--train-live` mode trains an imposed cue association; use
`train_onion.py` for the physical-outcome learning described in this guide.

**A useful next-session prompt:** “Continue from this folder's README. Find my
latest outcome-training checkpoint, summarize its results, and help me choose
the next experiment.”
