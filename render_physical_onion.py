"""Render diagnostic frames from the actual physical trial."""
import imageio.v3 as imageio
import mujoco as mj
import numpy as np
from onion_physics import PhysicalOnion, ROOT

scene = PhysicalOnion()
camera = mj.MjvCamera()
camera.lookat[:] = [*scene.path_xy,scene.board_z+0.3]
camera.distance,camera.azimuth,camera.elevation = 5.8,110,-30
out = ROOT/"data/experiments/physical_onion"
out.mkdir(parents=True,exist_ok=True)
with mj.Renderer(scene.model,height=600,width=800) as renderer:
    for k,t in enumerate(np.arange(0,scene.duration,scene.dt)):
        scene.step(t,True)
        if k in [350,750,1199]:
            renderer.update_scene(scene.data,camera=camera)
            imageio.imwrite(out/f"frame_{k}.png",renderer.render())
