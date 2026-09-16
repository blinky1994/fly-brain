"""Train chop/hold from physical outcomes, with an optional live viewer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np

from malecns_onion import BrainClient, ROOT
from onion_physics import PhysicalOnion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headless", action="store_true", help="Train without rendering or pacing")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--speed", type=float, default=4, help="Playback multiplier; 0 runs without deliberate waits")
    p.add_argument("--explore", type=float, default=0.35, help="Probability of trying the opposite decision")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--frozen", action="store_true", help="Control: execute trials without weight updates")
    p.add_argument("--no-cache", action="store_true", help="Recompute every neural response")
    p.add_argument("--exit-after-training", action="store_true")
    args = p.parse_args()
    if args.episodes < 1 or not np.isfinite(args.speed) or args.speed < 0 or not 0 <= args.explore <= 1:
        p.error("Use positive episodes, nonnegative speed, and exploration between 0 and 1")
    out = (args.output_dir or ROOT/"data/experiments/outcome_training"/str(time.time_ns())).resolve()
    if not out.is_relative_to(ROOT.resolve()) or (out.exists() and any(out.iterdir())):
        p.error("Output must be a new empty directory inside this project")
    out.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), output_dir=str(out), checkpoint=str(args.checkpoint) if args.checkpoint else None,
                  scope="Outcome-trained chop/hold; fixed leg trajectories; artificial signed plasticity",
                  reward_limits={"slip_mm":0.12,"blade_force_model_units":50},
                  cache="Deterministic cue response and eligibility, invalidated on any weight change")
    (out/"config.json").write_text(json.dumps(config,indent=2))
    print(f"Session: {out}",flush=True)
    print("Loading physical scene and neural network...",flush=True)
    scene = PhysicalOnion()
    brain = BrainClient(args.checkpoint)
    pool = ThreadPoolExecutor(max_workers=1)
    viewer = None
    started = time.perf_counter()
    results = []
    try:
        brain.request({"operation":"start_training","output_dir":str(out),
                       "from_scratch":args.checkpoint is None})
        if not args.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(scene.model,scene.data)
            viewer.cam.lookat[:] = [*scene.path_xy,scene.board_z+0.5]
            viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation = 6.5,125,-25

        def status(message, phase, episode):
            if viewer:
                viewer.set_texts((None,None,"Onion outcome training\nPhase\nTrial\nStatus\nSpeed\nModel",
                    f"\n{phase}\n{episode}\n{message}\n{'Maximum' if args.speed == 0 else str(args.speed)+'x'}\nLearned chop/hold; fixed leg movements"))
                viewer.sync()

        def request(payload):
            future = pool.submit(brain.request,payload)
            deadline = time.perf_counter()+180
            while not future.done():
                if viewer and not viewer.is_running():
                    raise InterruptedError("Viewer closed; last completed update saved")
                if time.perf_counter() > deadline:
                    raise TimeoutError("Brain request exceeded 180 seconds")
                if viewer:
                    viewer.sync()
                time.sleep(0.01)
            return future.result()

        def trial(offset, phase, episode, learn=False, explore=0):
            with viewer.lock() if viewer else nullcontext():
                scene.reset(offset)
            status("Computing neural decision...",phase,episode)
            decision = request({"operation":"outcome_decide","lateral_error_mm":abs(offset),
                                "explore":explore,"use_cache":not args.no_cache})
            status(f"{decision['action'].upper()} | {'exploration' if decision['exploratory'] else 'policy'} | spikes {decision['MBON11_spikes']}",phase,episode)
            ticks = np.arange(0,scene.duration,scene.dt)
            for first in range(0,len(ticks),20):
                if viewer and not viewer.is_running():
                    raise InterruptedError("Viewer closed; last completed update saved")
                tick_start = time.perf_counter()
                with viewer.lock() if viewer else nullcontext():
                    for t in ticks[first:first+20]:
                        scene.step(t,decision["action"] == "chop")
                if viewer:
                    viewer.sync()
                    if args.speed:
                        time.sleep(max(0,scene.dt*len(ticks[first:first+20])/args.speed-(time.perf_counter()-tick_start)))
            result = request({"operation":"observe_outcome","trial_id":decision["trial_id"],
                              "metrics":scene.metrics(),"learn":learn})
            result.update(phase=phase,offset_mm=offset,episode=episode)
            results.append(result)
            with (out/"trials.jsonl").open("a") as stream:
                stream.write(json.dumps(result)+"\n")
            print(json.dumps({k:v for k,v in result.items() if k != "physics"}),flush=True)
            status(f"Reward {result['reward']:+.0f}: {result['reward_reason']} | {result['connections_changed']} connections updated",phase,episode)
            return result

        offsets = [0,0.08,-0.08,0.5,0.6,-0.5]
        for i,offset in enumerate(offsets):
            trial(offset,"baseline",i+1)
        for i in range(args.episodes):
            trial(0 if i%2 == 0 else 0.6,"training",i+1,not args.frozen,args.explore)
        for i,offset in enumerate(offsets):
            trial(offset,"evaluation",i+1)
        def score(phase):
            selected = [r for r in results if r["phase"] == phase]
            return sum((r["reward"] == 1 if abs(r["offset_mm"]) <= 0.12 else r["action"] == "hold") for r in selected)
        report = {"baseline_correct":score("baseline"),"evaluation_correct":score("evaluation"),
                  "evaluation_trials":6,"wall_seconds":time.perf_counter()-started,
                  "training_episodes":args.episodes,"frozen":args.frozen,
                  "checkpoint":str(out/"latest.npz"),"scope":config["scope"]}
        (out/"report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        i = 0
        while viewer and viewer.is_running() and not args.exit_after_training:
            trial(0 if i%2 == 0 else 0.6,"complete - evaluation only",i+1)
            i += 1
    except InterruptedError as exc:
        print(str(exc),flush=True)
    finally:
        brain.close()
        pool.shutdown(wait=True,cancel_futures=True)
        if viewer:
            viewer.close()


if __name__ == "__main__":
    main()
