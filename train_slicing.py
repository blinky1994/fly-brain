"""Learn thin-slice motor settings from fresh physical trials in a live viewer."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import queue
import time

import numpy as np

from thin_onion import ThinOnion
from slicing_policy import SlicingPolicy,PROFILES,TRAIN_OFFSETS,slice_reward

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes",type=int,default=len(PROFILES)*len(TRAIN_OFFSETS))
    parser.add_argument("--slices",type=int,choices=[3,5],default=3,help="Start with three; five is a harder curriculum")
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--speed",type=float,default=1,help="Body playback speed; 0 is maximum")
    parser.add_argument("--checkpoint",type=Path,help="Slicing latest.json, not a neural NPZ")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--seed",type=int,default=29)
    parser.add_argument("--continuous",action="store_true",help="Repeat train/evaluate blocks until viewer closes")
    parser.add_argument("--frozen",action="store_true",help="Keep the starting motor policy fixed")
    args=parser.parse_args()
    if args.episodes<1 or not np.isfinite(args.speed) or args.speed<0:
        parser.error("Use positive episodes and nonnegative speed")
    if args.continuous and args.headless:
        parser.error("Continuous mode requires a viewer so closing it stops training")
    out=(args.output_dir or ROOT/"data/experiments/slicing_training"/str(time.time_ns())).resolve()
    if not out.is_relative_to(ROOT) or (out.exists() and any(out.iterdir())):
        parser.error("Use a new empty output folder inside this project")
    out.mkdir(parents=True,exist_ok=True)
    policy=SlicingPolicy(args.checkpoint,args.seed,args.slices)
    policy.save(out/"latest.json")
    config={**vars(args),"checkpoint":str(args.checkpoint) if args.checkpoint else None,"output_dir":str(out),
            "scope":"Outcome-trained motor profiles; separate from MaleCNS synaptic learning",
            "thickness_mm":.05,"profiles":PROFILES,"training_offsets":TRAIN_OFFSETS}
    (out/"config.json").write_text(json.dumps(config,indent=2))
    print(f"Session: {out}",flush=True)
    print("Building slicing scene...",flush=True)
    scene=ThinOnion(args.slices)
    viewer=None
    keys=queue.SimpleQueue()
    results=[]
    started=time.perf_counter()
    try:
        if not args.headless:
            import mujoco.viewer
            viewer=mujoco.viewer.launch_passive(scene.model,scene.data,key_callback=keys.put)
            viewer.cam.lookat[:]=[*scene.path_xy,scene.board_z+.18]
            viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation=2.8,135,-30

        def status(phase,episode,action,explore,reward=None):
            if viewer:
                state=scene.metrics()
                viewer.set_texts((None,None,"Thin-slice motor training\nPhase\nEpisode\nAction\nStroke\nResult\nControls",
                    f"\n{phase}\n{episode}\n{PROFILES[action]['name']} ({'exploration' if explore else 'policy'})"
                    f"\n{scene.stroke_index+1}/{args.slices} | 0.05 mm preset spacing"
                    f"\n{state['clean_slices']}/{args.slices} clean | {'running' if reward is None else f'reward {reward:.3f}'}"
                    "\n1 slow / 2 normal / 3 fast / 0 maximum / close to stop"))

        def trial(phase,episode,offset,nudge,learn):
            if viewer and not viewer.is_running():
                raise InterruptedError
            action,explore=policy.choose(learn=learn)
            context=policy.context_index(action) if learn else None
            if learn:
                offset=TRAIN_OFFSETS[context]
            # Reset and IK planning do not advance physics. Results always come
            # from fresh simulation; only geometric pose solutions are cached.
            with viewer.lock() if viewer else nullcontext():
                scene.reset(offset=offset,profile=PROFILES[action],nudge=nudge)
            for t in np.arange(0,scene.duration,scene.dt):
                if viewer and not viewer.is_running():
                    raise InterruptedError
                tick=time.perf_counter()
                with viewer.lock() if viewer else nullcontext():
                    scene.step(t)
                if scene.failure_reason is not None:
                    break
                if viewer:
                    while not keys.empty():
                        key=keys.get()
                        if key in [ord('0'),ord('1'),ord('2'),ord('3')]:
                            args.speed={ord('0'):0,ord('1'):.5,ord('2'):1,ord('3'):4}[key]
                    if round(t/scene.dt)%10==0:
                        status(phase,episode,action,explore)
                        viewer.sync()
                    if args.speed:
                        time.sleep(max(0,scene.dt/args.speed-(time.perf_counter()-tick)))
            metrics=scene.metrics()
            reward=slice_reward(metrics)
            if learn:
                policy.update(action,reward,context)
                policy.save(out/"latest.json")
            result=dict(phase=phase,episode=episode,action=action,profile=PROFILES[action],
                        exploratory=explore,learning=learn,reward=reward,offset_mm=offset,nudge=nudge,metrics=metrics)
            results.append(result)
            with (out/"trials.jsonl").open("a") as stream:
                stream.write(json.dumps(result)+"\n")
            print(f"{phase} {episode}: {PROFILES[action]['name']} | clean {metrics['clean_slices']}/{args.slices} | "
                  f"detached {metrics['slices_detached']}/{args.slices} | reward {reward:.3f} | learning {learn}",flush=True)
            status(phase,episode,action,explore,reward)
            if viewer:
                until=time.perf_counter()+.8
                while viewer.is_running() and time.perf_counter()<until:
                    viewer.sync()
                    time.sleep(.02)

        # These positions are distinct from the training set, but remain tiny
        # variations of one engineered scene, not evidence of general skill.
        evaluation=[(-.012,0),(.012,0),(0,.015)]
        for i,(offset,nudge) in enumerate(evaluation):
            trial("baseline",i+1,offset,nudge,False)
        block=0
        while True:
            for i in range(args.episodes):
                offset=[0,-.006,.006][i%3]
                trial("training",block*args.episodes+i+1,offset,0,not args.frozen)
            for i,(offset,nudge) in enumerate(evaluation):
                trial("evaluation",block*len(evaluation)+i+1,offset,nudge,False)
            def summary(rows):
                return dict(trials=len(rows),mean_reward=float(np.mean([r["reward"] for r in rows])),
                    clean_slices=sum(r["metrics"]["clean_slices"] for r in rows),
                    target_slices=args.slices*len(rows),complete_clean_sequences=sum(r["metrics"]["clean_slices"]==args.slices for r in rows))
            report=dict(baseline=summary([r for r in results if r["phase"]=="baseline"]),
                        evaluation=summary([r for r in results if r["phase"]=="evaluation"][-3:]),
                        trained_episodes=int(policy.counts.sum()),blocks_completed=block+1,
                        best_profile=PROFILES[policy.choose(False)[0]],wall_seconds=time.perf_counter()-started,
                        scope=config["scope"],checkpoint=str(out/"latest.json"),frozen=args.frozen)
            report["training_position_counts"]=policy.context_counts.tolist()
            (out/"report.json").write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            block+=1
            if not args.continuous:
                break
            results[:]=[r for r in results if r["phase"]=="baseline"]
    except (InterruptedError,KeyboardInterrupt):
        print("Stopped; the last completed training episode is saved.",flush=True)
    finally:
        if viewer:
            viewer.close()


if __name__=="__main__":
    main()
