"""Physical slicing controls and rendered evidence for a motor checkpoint."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import mujoco as mj
import imageio.v3 as imageio
from thin_onion import ThinOnion
from slicing_policy import SlicingPolicy,PROFILES,slice_reward


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output-dir",type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    out=(args.output_dir or root/"data/experiments/slicing_checks"/str(time.time_ns())).resolve()
    if not out.is_relative_to(root) or (out.exists() and any(out.iterdir())):
        p.error("Use a new empty output directory inside this project")
    out.mkdir(parents=True,exist_ok=True)
    policy=SlicingPolicy(args.checkpoint)
    action=policy.choose(False)[0]
    scene=ThinOnion()
    camera=mj.MjvCamera()
    camera.lookat[:]=[*scene.path_xy,scene.board_z+.16]
    camera.distance,camera.azimuth,camera.elevation=1.65,135,-35
    report={"checkpoint":str(args.checkpoint),"profile":PROFILES[action],"controls":{}}
    for mode in ["policy","no_chop","locked_seams","no_hold"]:
        scene.reset(profile=PROFILES[action])
        scene.fracture_enabled=mode!="locked_seams"
        scene.hold_enabled=mode!="no_hold"
        if mode=="policy":
            with mj.Renderer(scene.model,height=700,width=1000) as renderer:
                renderer.update_scene(scene.data,camera=camera)
                imageio.imwrite(out/"before.png",renderer.render())
        for t in np.arange(0,scene.duration,scene.dt):
            scene.step(t,chop=mode!="no_chop")
            if scene.failure_reason:
                break
        metrics=scene.metrics()
        report["controls"][mode]=dict(reward=slice_reward(metrics),**metrics)
        if mode=="policy":
            with mj.Renderer(scene.model,height=700,width=1000) as renderer:
                renderer.update_scene(scene.data,camera=camera)
                imageio.imwrite(out/"after.png",renderer.render())
        print(mode,json.dumps(metrics),flush=True)
        (out/"report.json").write_text(json.dumps(report,indent=2))
    results=report["controls"]
    assert results["policy"]["clean_slices"]==3,"Policy did not complete three clean slices"
    assert results["policy"]["blade_foot_force"]<=.001
    assert results["no_chop"]["seams_broken"]==0,"Fracture without a cutting stroke"
    assert results["locked_seams"]["seams_broken"]==0,"Locked seam fractured"
    assert results["no_hold"]["clean_slices"]==0,"Unheld cuts counted as clean"
    print("Physical controls passed:",out,flush=True)


if __name__=="__main__":
    main()
