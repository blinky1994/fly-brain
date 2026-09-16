# MaleCNS neural experiments

These scripts use the local MaleCNS v1.0 import, not the older FlyBody cooking
controller. Run from this folder in Git Bash with the Python 3.11 environment:

```bash
source .venv/Scripts/activate
python malecns_probe.py
python malecns_conditioning.py
```

Outputs are under `data/experiments/neural_probe` and
`data/experiments/conditioning`. Re-running replaces these experiment outputs.
The original downloaded data and processed anatomical counts are not modified.

## What runs

All 166,700 selected neuron records and 25,582,938 anatomical directed connections
are loaded. Point-neuron dynamics are an experimental, uniform leaky
integrate-and-fire approximation. The probe contrasts no input with direct
stimulation of 100 Kenyon cells with existing connections to MBON11. This bypasses
sensory encoding entirely.

The conditioning experiment makes disjoint 100-KC cues A and B. It compares six
training pairs in each of three conditions: A paired with imposed PPL101 drive,
the same protocol with plasticity frozen, and B paired instead of A. Every trial
starts from resting fast neural state; only trained synaptic efficacies persist.
Before and after tests have no imposed reinforcement and no plasticity.

Only the 4,184 existing KC-to-MBON11 connections are eligible to change. The
experiment imposes a dopamine-gated depression rule with a 1-second eligibility
trace, eta 0.05, an efficacy floor of 10%, and a 1.8-ms modulatory delay. PPL101
modulation is assigned to the same-side MBON11 inputs. These are chosen modeling
assumptions, not a learning rule contained in the connectome. Internal efficacy
changes are saved with biological neuron IDs in the NPZ files.

## Limits of interpretation

Fast transmission is positive for acetylcholine and negative for GABA, glutamate,
and histamine. This ignores receptor-specific effects. Missing/unclear labels,
dopamine, octopamine, and serotonin have zero fast effect in this initial model;
their anatomical connections are retained. PPL101 has only the explicitly modeled
modulatory effect in the conditioning experiment. No background noise or tonic
drive is used. Parameters and omitted-label counts are recorded in reports.

A decrease in the paired cue's MBON11 response, a stable frozen control, and a
reversal when B is paired support cue-specific plasticity in this constructed
circuit assay. They do not establish that the model perceives odors, makes a
learned choice, controls a body, or reproduces biological learning. The first run
uses one deterministic cue selection; robustness across selections, strengths,
delays, and network assumptions still needs testing. No cooking capability or
behavioral-learning success should be inferred.

Background references:

- Official data: https://male-cns.janelia.org/download/
- Related experimental dynamics: https://github.com/nftechie/doomfly/blob/main/doom/kernel.cpp
- Related conditioning approach and its limits: https://github.com/nftechie/doomfly/blob/main/doom_learning/README.md

Our scripts are a separate experimental implementation, not an execution or
reproduction of DOOMFLY's validated test suite.

## Follow-up: external reinforcement gate

The first run produced PPL101 spikes for both cues without imposed reinforcement.
Consequently, using every PPL101 spike as a learning signal depressed both cue
responses. `malecns_reward_gate.py` tests one explicit engineering change: only
PPL101 spikes during an externally supplied reinforcement window are accepted
for synaptic updates. Neural firing and anatomical connections are unchanged.
This is an externally gated learning model, not a discovery about fly dopamine.

```bash
python malecns_reward_gate.py
```

The run tests three preselected cue-assignment seeds (7, 23, 101). Each has A-paired,
B-paired, frozen-plasticity, and no-imposed-reinforcement conditions. Learning rate,
trace duration, weight floor, trial timing, and six training pairs stay unchanged.
The stated success criterion is at least 50% suppression for the paired cue and
at least 80% retention of the unpaired cue's original MBON11 spike response, with
both control conditions exactly unchanged. All evaluation uses reinforcement and
plasticity off. A zero baseline response makes that pairing fail the criterion.
Results and checkpoints go to `data/experiments/conditioning_reward_gate`, leaving
the first failed experiment's outputs intact. Re-running replaces this follow-up's
outputs. These tests still measure a neural response, not a learned motor action.

## More rounds, focused runs, and continuing a checkpoint

Close the live viewer first to avoid competing for CPU time. To continue the
demo's checkpoint for 20 additional A+B rounds (40 cue presentations):

```bash
.venv/Scripts/python.exe malecns_reward_gate.py --seeds 7 --rounds 20 --train-only --resume data/experiments/conditioning_reward_gate/seed7_paired_A.npz --output-dir data/experiments/onion_more
```

`--train-only` runs just A-pairing instead of the four conditions. Together with
`--seeds 7`, this runs one training condition instead of twelve, while retaining
before/after neural response checks. It does not accelerate individual neural
steps, and omitting controls must not be reported as full validation. The report
records that controls were skipped. The original data, timestep, neuron count,
and learning rate are unchanged. This implementation runs on CPU.

Load the resulting checkpoint explicitly in a new viewer process:

```bash
.venv-body/Scripts/python.exe malecns_onion.py watch --physical --checkpoint data/experiments/onion_more/seed7_paired_A.npz
```

For another session, use that new NPZ as `--resume` and choose a new output
directory. Without `--resume`, each run starts from original anatomical
efficacies. More rounds do not guarantee improvements: this simple depression
rule has a floor, and the demonstrated two-cue response is already successful.
Watching never updates weights. Online training from physical cuts and learning
the left-foot holding skill remain separate implementation work.
