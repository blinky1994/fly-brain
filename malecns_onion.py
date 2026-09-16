"""Live fly body using fresh MaleCNS neural decisions through an isolated worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import time

import mujoco as mj
import numpy as np

from onion_scene import OnionScene

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/experiments/onion_bridge"


class BrainClient:
    def __init__(self,checkpoint=None):
        OUT.mkdir(parents=True, exist_ok=True)
        python = ROOT / ".venv/Scripts/python.exe"
        self.log_path = OUT/f"brain_worker_{os.getpid()}_{time.time_ns()}.log"
        self.log = self.log_path.open("w")
        env = os.environ.copy()
        if checkpoint is not None:
            env["MALECNS_ONION_CHECKPOINT"] = str(checkpoint.resolve())
        else:
            env.pop("MALECNS_ONION_CHECKPOINT",None)
        self.process = subprocess.Popen(
            [str(python), "-u", str(ROOT/"malecns_onion_brain.py")],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1,env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            ready = self.read()
            if not ready.get("ready"):
                raise RuntimeError(f"Brain failed to load: {ready}")
        except BaseException:
            self.close()
            raise

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Brain worker stopped. See {self.log_path}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def decide(self, error, mode):
        return self.request({"lateral_error_mm":float(error),"mode":mode})

    def request(self,payload):
        self.process.stdin.write(json.dumps(payload)+"\n")
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


def benchmark(scene,brain):
    results = []
    # Small position variations; these still collapse to two encoded cue classes.
    for mode in ["untrained","trained","wrong_pairing"]:
        for offset in [0,0.08,-0.08,0.5,0.6,-0.5]:
            scene.reset(offset)
            error = float(np.linalg.norm(scene.onion_xy-scene.path_xy))
            result = brain.decide(error,mode)
            cut = scene.execute(result["action"] == "chop")
            aligned = abs(offset) <= 0.12
            correct = (cut if aligned else result["action"] == "hold")
            if aligned and hasattr(scene,"metrics"):
                physics = scene.metrics()
                correct = (correct and physics["first_pad_contact_s"] is not None
                           and physics["first_pad_contact_s"] < physics["cut_time_s"]
                           and physics["pad_force_at_fracture_model_units"] > 0.001
                           and physics["knife_pad_force_model_units"] == 0)
            result.update({"offset_mm":offset,"cut":bool(cut),"correct":bool(correct)})
            if hasattr(scene,"metrics"):
                result["physics"] = scene.metrics()
            results.append(result)
            print(json.dumps(result),flush=True)
    report = {"scope":"Transfer of preconditioned neural cues to a fixed motor primitive",
              "training_in_kitchen":False,"visual_perception":False,
              "encoding":"Geometric alignment error <=0.12 mm selects KC cue A, otherwise B",
              "decoder":"MBON11 total spikes <=1 -> chop; otherwise hold",
              "decoder_selection":"Chosen using earlier conditioning results, not an independent held-out decoder",
              "physics":("Two free onion halves, breakable seam, frictional left pad; fixed motor skill"
                         if hasattr(scene,"metrics") else "Supported body; attached knife; geometric cut event; no food fracture"),
              "trials":results,
              "correct_by_mode":{mode:sum(r["correct"] for r in results if r["mode"] == mode)
                                   for mode in ["untrained","trained","wrong_pairing"]},
              "trials_per_mode":6}
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("Correct trials:",report["correct_by_mode"],flush=True)
    if report["correct_by_mode"]["trained"] != 6:
        raise RuntimeError("Trained bridge failed a trial; see saved report.")
    if report["correct_by_mode"]["trained"] <= report["correct_by_mode"]["untrained"]:
        raise RuntimeError("No improvement over untrained control")


def watch(scene,brain,mode,max_trials=0,live_rounds=0,training_dir=None,from_scratch=False):
    import mujoco.viewer
    keys = queue.SimpleQueue()
    executor = ThreadPoolExecutor(max_workers=1)
    forced_offset = None
    trial_number = 0
    completed_rounds = 0
    last_changes = 0
    if live_rounds:
        mode = "trained"
        print(json.dumps(brain.request({"operation":"start_training",
            "output_dir":str(training_dir),"from_scratch":from_scratch})),flush=True)
    try:
        with mujoco.viewer.launch_passive(scene.model,scene.data,key_callback=keys.put) as viewer:
            viewer.cam.lookat[:] = [*scene.path_xy,scene.board_z+0.5]
            viewer.cam.distance = 6.5
            viewer.cam.azimuth = 125
            viewer.cam.elevation = -25

            def overlay(message,offset):
                condition = "ALIGNED" if abs(offset) <= 0.12 else "MISALIGNED"
                physics = (f"Left pad: {'contact' if scene.current_pad_force > 0.001 else 'no contact'} | seam: {'broken' if scene.cut else 'intact'}"
                           if hasattr(scene,"metrics") else "Supported fly, attached knife, simplified cut")
                training = (f"Cue training {completed_rounds}/{live_rounds}; last update: {last_changes} connections"
                            if live_rounds else "Training OFF")
                if live_rounds and completed_rounds == live_rounds:
                    training = f"Training complete: {live_rounds} rounds saved. Evaluation only."
                controls = ("A aligned / M misaligned / X alternate" if live_rounds
                            else "T trained / U untrained / W wrong pairing")
                viewer.set_texts((None,None,
                    "MaleCNS onion experiment\nBrain\nOnion\nStatus\nTraining\nControls\n\nModel",
                    f"\n{'live learner' if live_rounds else mode}\n{condition}\n{message}\n{training}\n{controls}\n"
                    f"{'Cue pairing trains; body trials evaluate' if live_rounds else 'A aligned / M misaligned / X alternate (next trial)'}\n{physics}"))

            def wait_for(request):
                deadline = time.perf_counter()+120
                while not request.done():
                    if not viewer.is_running():
                        return None
                    if time.perf_counter() > deadline:
                        raise TimeoutError("Neural operation exceeded two minutes")
                    viewer.sync()
                    time.sleep(0.02)
                return request.result()

            while viewer.is_running():
                while not keys.empty():
                    key = keys.get()
                    if not live_rounds and key in [ord("T"),ord("U"),ord("W")]:
                        mode = {ord("T"):"trained",ord("U"):"untrained",ord("W"):"wrong_pairing"}[key]
                    elif key == ord("A"):
                        forced_offset = 0
                    elif key == ord("M"):
                        forced_offset = 0.6
                    elif key == ord("X"):
                        forced_offset = None
                offset = forced_offset if forced_offset is not None else (0 if trial_number%2 == 0 else 0.6)
                # Show two baseline/body trials, then train between each pair.
                if live_rounds and trial_number > 0 and trial_number%2 == 0 and completed_rounds < live_rounds:
                    overlay(f"Training round {completed_rounds+1}: pairing cue A...",offset)
                    update = wait_for(executor.submit(brain.request,{"operation":"train_round"}))
                    if update is None:
                        return
                    completed_rounds = update["round"]
                    last_changes = update["connections_changed_this_round"]
                    print(json.dumps(update),flush=True)
                with viewer.lock():
                    scene.reset(offset)
                overlay("Computing neural response...",offset)
                request = executor.submit(brain.decide,
                    np.linalg.norm(scene.onion_xy-scene.path_xy),mode)
                result = wait_for(request)
                if result is None:
                    return
                chop = result["action"] == "chop"
                overlay(f"{result['action'].upper()} | MBON11 spikes: {result['MBON11_spikes']}",offset)
                ticks = np.arange(0,scene.duration,scene.dt)
                for first in range(0,len(ticks),10):
                    if not viewer.is_running():
                        return
                    start = time.perf_counter()
                    with viewer.lock():
                        for t in ticks[first:first+10]:
                            scene.step(t,chop)
                    overlay(f"{result['action'].upper()} | MBON11 spikes: {result['MBON11_spikes']}",offset)
                    viewer.sync()
                    time.sleep(max(0,scene.dt*20-(time.perf_counter()-start)))
                status = "CUT" if scene.cut else ("HELD" if not chop else "MISSED")
                overlay(f"{status} | MBON11 spikes: {result['MBON11_spikes']}",offset)
                result.update({"cut":bool(scene.cut),"trial":trial_number})
                if live_rounds:
                    result["completed_training_rounds"] = completed_rounds
                    with (training_dir/"body_trials.jsonl").open("a") as log:
                        log.write(json.dumps(result)+"\n")
                print(json.dumps(result),flush=True)
                until = time.perf_counter()+1
                while viewer.is_running() and time.perf_counter() < until:
                    viewer.sync()
                    time.sleep(0.02)
                trial_number += 1
                if max_trials and trial_number >= max_trials:
                    return
    finally:
        brain.close()
        executor.shutdown(wait=True,cancel_futures=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["watch","check"])
    parser.add_argument("--mode",choices=["trained","untrained","wrong_pairing"],default="trained")
    parser.add_argument("--trials",type=int,default=0,help="Close after N trials; zero repeats until closed")
    parser.add_argument("--physical",action="store_true",help="Physical onion and left-foot holding")
    parser.add_argument("--checkpoint",type=Path,help="Seed-7 checkpoint to use for trained mode")
    parser.add_argument("--train-live",action="store_true",help="Cue training between displayed body trials")
    parser.add_argument("--rounds",type=int,default=6,help="Live training rounds; then evaluation continues")
    parser.add_argument("--training-dir",type=Path,help="New empty directory for live checkpoints")
    args = parser.parse_args()
    if args.train_live and (args.command != "watch" or args.rounds < 1):
        parser.error("Live training requires watch and positive --rounds")
    training_dir = None
    if args.train_live:
        training_dir = (args.training_dir or ROOT/"data/experiments/live_training"/str(time.time_ns())).resolve()
        if not training_dir.is_relative_to(ROOT.resolve()):
            parser.error("Training directory must be inside this project")
        if training_dir.exists() and any(training_dir.iterdir()):
            parser.error("Use a new empty --training-dir to preserve previous runs")
    print("Loading fly body and MaleCNS brain...",flush=True)
    if args.physical:
        from onion_physics import PhysicalOnion
        scene = PhysicalOnion()
        OUT = ROOT / "data/experiments/physical_onion_bridge"
    else:
        scene = OnionScene()
    brain = BrainClient(args.checkpoint)
    try:
        if args.command == "check":
            benchmark(scene,brain)
        else:
            watch(scene,brain,args.mode,args.trials,args.rounds if args.train_live else 0,
                  training_dir,from_scratch=args.checkpoint is None)
    finally:
        brain.close()


if __name__ == "__main__":
    main()
