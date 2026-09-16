"""JSON-lines bridge: full MaleCNS dynamics -> engineered chop/hold readout."""
import json
import sys
import os
from pathlib import Path

import numpy as np

from malecns_conditioning import Assay
from malecns_probe import ROOT
from onion_outcome import cutting_reward, update_weights


class OnionBrain:
    def __init__(self):
        self.assay = Assay(seed=7, reward_gate=True)
        a = self.assay
        self.saved = {}
        for mode, arm in [("trained", "paired_A"), ("wrong_pairing", "paired_B")]:
            path = ROOT / f"data/experiments/conditioning_reward_gate/seed7_{arm}.npz"
            if mode == "trained" and os.environ.get("MALECNS_ONION_CHECKPOINT"):
                path = Path(os.environ["MALECNS_ONION_CHECKPOINT"])
            with np.load(path, allow_pickle=False) as data:
                assert np.array_equal(data["pre_body"], a.brain.neurons.bodyId.iloc[a.pre])
                assert np.array_equal(data["post_body"], a.brain.neurons.bodyId.iloc[a.post])
                assert np.allclose(data["original"], a.original)
                assert np.array_equal(data["cue_A"], a.brain.neurons.bodyId.iloc[a.cues["A"]])
                assert np.array_equal(data["cue_B"], a.brain.neurons.bodyId.iloc[a.cues["B"]])
                self.saved[mode] = data["trained"].copy()
        self.saved["untrained"] = a.original.copy()
        self.training_rounds = 0
        self.training_dir = None
        self.response_cache = {}
        self.pending_outcome = None
        self.outcome_id = 0
        self.rng = np.random.default_rng(19)

    def start_training(self, output_dir, from_scratch):
        directory = Path(output_dir).resolve()
        if not directory.is_relative_to(ROOT.resolve()):
            raise ValueError("Live checkpoints must be inside this project")
        directory.mkdir(parents=True,exist_ok=True)
        self.training_dir = directory
        self.training_rounds = 0
        self.pending_outcome = None
        if from_scratch:
            self.saved["trained"] = self.assay.original.copy()
        self.response_cache.clear()
        self.save_training()
        return {"training_ready":True,"rounds":0,"from_scratch":from_scratch,
                "checkpoint":str(directory/"latest.npz")}

    def save_training(self):
        a = self.assay
        target = self.training_dir/"latest.npz"
        temporary = self.training_dir/"latest.partial"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,pre_body=a.brain.neurons.bodyId.iloc[a.pre],
                post_body=a.brain.neurons.bodyId.iloc[a.post],original=a.original,
                trained=self.saved["trained"],cue_A=a.brain.neurons.bodyId.iloc[a.cues["A"]],
                cue_B=a.brain.neurons.bodyId.iloc[a.cues["B"]],
                live_session_rounds=self.training_rounds)
        temporary.replace(target)

    def train_round(self):
        if self.training_dir is None:
            raise RuntimeError("Start a live training session first")
        a = self.assay
        previous = self.saved["trained"].copy()
        a.brain.weights.data[a.positions] = previous
        a.trial("A",reinforce=True,plastic=True)
        a.trial("B",reinforce=False,plastic=True)
        self.saved["trained"] = a.brain.weights.data[a.positions].copy()
        self.response_cache.clear()
        self.training_rounds += 1
        self.save_training()
        result = {"round":self.training_rounds,
                  "connections_changed_this_round":int(np.count_nonzero(previous != self.saved["trained"])),
                  "checkpoint":str(self.training_dir/"latest.npz"),
                  "reinforcement_source":"imposed cue A pairing; not cutting outcome"}
        with (self.training_dir/"rounds.jsonl").open("a") as log:
            log.write(json.dumps(result)+"\n")
        return result

    def decide(self, lateral_error_mm, mode="trained"):
        if mode not in self.saved or not np.isfinite(lateral_error_mm) or lateral_error_mm < 0:
            raise ValueError("Invalid mode or alignment measurement")
        a = self.assay
        a.brain.weights.data[a.positions] = self.saved[mode]
        # Explicit engineered sensor encoding; no visual perception claim.
        cue = "A" if lateral_error_mm <= 0.12 else "B"
        response = a.trial(cue, reinforce=False, plastic=False)
        spikes = response["MBON11_spikes"]
        # Fixed, engineered decoder; chosen using the preceding conditioning assay.
        return {"action":"chop" if spikes <= 1 else "hold",
                "MBON11_spikes":spikes, "cue":cue, "mode":mode,
                "lateral_error_mm":lateral_error_mm,
                "learning_during_decision":False}

    def outcome_decide(self, error, explore=0.35, use_cache=True):
        if self.training_dir is None or self.pending_outcome is not None:
            raise ValueError("Start training and finish the previous outcome first")
        if not np.isfinite(error) or error < 0 or not 0 <= explore <= 1:
            raise ValueError("Invalid decision parameters")
        cue = "A" if error <= 0.12 else "B"
        a = self.assay
        cached = use_cache and cue in self.response_cache
        if cached:
            response, trace = self.response_cache[cue]
        else:
            a.brain.weights.data[a.positions] = self.saved["trained"]
            response = a.trial(cue, capture_eligibility=True)
            trace = a.last_eligibility.copy()
            if use_cache:
                self.response_cache[cue] = (response, trace)
        policy = "chop" if response["MBON11_spikes"] <= 1 else "hold"
        exploratory = bool(self.rng.random() < explore)
        action = ("hold" if policy == "chop" else "chop") if exploratory else policy
        self.outcome_id += 1
        result = {"trial_id":self.outcome_id, "cue":cue, "action":action,
                  "policy_action":policy, "exploratory":exploratory,
                  "MBON11_spikes":response["MBON11_spikes"], "cached_response":bool(cached)}
        self.pending_outcome = (result, trace)
        return result

    def observe_outcome(self, trial_id, metrics, learn=True):
        if self.pending_outcome is None or trial_id != self.pending_outcome[0]["trial_id"]:
            raise ValueError("Outcome does not match pending decision")
        decision, trace = self.pending_outcome
        reward, reason = cutting_reward(decision["action"], metrics)
        previous = self.saved["trained"]
        updated = (update_weights(previous, self.assay.original, trace, reward, decision["action"])
                   if learn else previous.copy())
        changed = int(np.count_nonzero(updated != previous))
        self.saved["trained"] = updated
        if changed:
            self.response_cache.clear()
        if learn:
            self.training_rounds += 1
            self.save_training()
        result = dict(decision, reward=reward, reward_reason=reason, learning=learn,
                      connections_changed=changed, physics=metrics)
        with (self.training_dir/"outcomes.jsonl").open("a") as log:
            log.write(json.dumps(result)+"\n")
        self.pending_outcome = None
        return result


def main():
    try:
        brain = OnionBrain()
        print(json.dumps({"ready":True, "neurons":brain.assay.brain.n}), flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                operation = request.get("operation","decide")
                if operation == "start_training":
                    result = brain.start_training(request["output_dir"],bool(request.get("from_scratch",False)))
                elif operation == "train_round":
                    result = brain.train_round()
                elif operation == "decide":
                    result = brain.decide(float(request["lateral_error_mm"]), request.get("mode", "trained"))
                elif operation == "outcome_decide":
                    result = brain.outcome_decide(float(request["lateral_error_mm"]),
                        float(request.get("explore",0.35)),bool(request.get("use_cache",True)))
                elif operation == "observe_outcome":
                    result = brain.observe_outcome(request["trial_id"],request["metrics"],
                                                   bool(request.get("learn",True)))
                else:
                    raise ValueError("Unknown worker operation")
            except Exception as exc:
                result = {"error":f"{type(exc).__name__}: {exc}"}
            print(json.dumps(result), flush=True)
    except Exception as exc:
        print(json.dumps({"error":f"{type(exc).__name__}: {exc}"}), flush=True)
        raise


if __name__ == "__main__":
    main()
