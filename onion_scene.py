"""Supported FlyBody, attached knife, fixed motor skill, simplified onion cutting."""
from pathlib import Path
import os

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares

from flygym.compose import ActuatorType, KinematicPosePreset, TetheredWorld
from flygym.compose.fly import FlyBody
from flygym.flybody import FlyBodyActuatedDOFPreset, FlyBodyAxisOrder
from flygym.flybody import FlyBodyJointPreset, FlyBodySkeleton
from flygym.utils.math import Rotation3D


class OnionScene:
    def __init__(self):
        local_assets = Path(__file__).resolve().parent / "data"
        if (local_assets/"flybody_fullsize_meshes_20260623a").is_dir():
            os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", str(local_assets))
        fly = FlyBody(name="chef")
        skeleton = FlyBodySkeleton(axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
                                   joint_preset=FlyBodyJointPreset.LEGS_ONLY)
        fly.add_joints(skeleton, KinematicPosePreset.FLYBODY_NEUTRAL)
        dofs = skeleton.get_actuated_dofs_from_preset(FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        fly.add_actuators(dofs, ActuatorType.POSITION, kp=100)
        fly.colorize()
        world = TetheredWorld()
        self.world = world
        world.add_fly(fly, [0, 0, 1.5], Rotation3D("quat", [1, 0, 0, 0]))
        spec = world.mjcf_root
        for geom in spec.geoms:
            geom.contype = geom.conaffinity = 0
        model, data = world.compile()
        mj.mj_resetDataKeyframe(model, data, model.key("neutral").id)
        mj.mj_forward(model, data)
        foot_id = model.body("chef/rf_tarsus1").id
        inverse = data.xquat[foot_id].copy()
        inverse[1:] *= -1
        foot = next(b for b in spec.bodies if b.name == "chef/rf_tarsus1")
        knife = foot.add_body(name="knife", quat=inverse)
        knife.add_geom(name="blade", type=mj.mjtGeom.mjGEOM_BOX,
            pos=[0.3,0,0], size=[0.3,0.025,0.16], mass=1e-7,
            rgba=[0.75,0.82,0.9,1], contype=1, conaffinity=1)
        knife.add_geom(name="handle", type=mj.mjtGeom.mjGEOM_CAPSULE,
            fromto=[-0.2,0,0,0,0,0], size=[0.06,0,0], mass=1e-7,
            rgba=[0.15,0.08,0.04,1], contype=0, conaffinity=0)
        knife.add_site(name="edge", pos=[0.3,0,-0.16], size=[0.025]*3,
                       rgba=[1,0.7,0,1])
        center = data.xpos[foot_id]+[0.3,0,-0.16]
        self.path_xy = center[:2].copy()
        self.board_z = float(center[2]-0.2)
        self.top_z = self.board_z+0.32
        wb = spec.worldbody

        def box(name, pos, size, color, contact=False):
            wb.add_geom(name=name, type=mj.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                        rgba=color, contype=int(contact), conaffinity=int(contact))

        x,y = self.path_xy
        box("floor", [0,0,-0.1], [8,8,0.1], [0.22,0.27,0.3,1])
        box("table", [x,y+0.2,self.board_z-0.18], [1,1,0.1], [0.4,0.22,0.1,1])
        box("board", [x,y+0.2,self.board_z-0.04], [0.9,0.9,0.04], [0.75,0.56,0.3,1], True)
        for i,(dx,dy) in enumerate([(-0.8,-0.6),(-0.8,1),(0.8,-0.6),(0.8,1)]):
            h = max(0.1,self.board_z-0.28)
            box(f"leg{i}", [x+dx,y+dy,h/2], [0.055,0.055,h/2], [0.4,0.22,0.1,1])
        torso = data.xpos[model.body("chef/c_thorax").id].copy()
        box("support", [torso[0]-0.15,torso[1],torso[2]/2], [0.1,0.1,torso[2]/2], [0.45,0.5,0.55,1])
        for name, visible in [("onion",True),("piece_left",False),("piece_right",False)]:
            wb.add_geom(name=name, type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                pos=[x,y,self.board_z+0.16], size=[0.24,0.24 if visible else 0.11,0.16],
                rgba=[0.65,0.28,0.5,float(visible)], contype=0, conaffinity=0)
        self.model, self.data = world.compile()
        self.key = self.model.key("neutral").id
        self.edge = self.model.site("edge").id
        self.indices = np.array([i for i in range(self.model.nu)
                                 if "rf_" in self.model.actuator(i).name])
        self.joints = self.model.actuator_trnid[self.indices,0]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.substeps = max(1,round(0.002/self.model.opt.timestep))
        self.dt = self.substeps*self.model.opt.timestep
        self.duration = 0.9
        self.reset(0)
        self.neutral = self.data.qpos[self.qadr].copy()
        # A deterministic inverse-kinematics motor primitive, NOT learned by MaleCNS.
        self.raised = self.solve_pose(np.r_[self.path_xy,self.top_z+0.16])
        self.lowered = self.solve_pose(np.r_[self.path_xy,self.board_z+0.025])
        self.reset(0)

    def solve_pose(self, target):
        lower = self.model.jnt_range[self.joints,0]+0.001
        upper = self.model.jnt_range[self.joints,1]-0.001
        probe = mj.MjData(self.model)
        mj.mj_resetDataKeyframe(self.model,probe,self.key)
        def residual(q):
            probe.qpos[self.qadr] = q
            mj.mj_forward(self.model,probe)
            rot = probe.site_xmat[self.edge].reshape(3,3)
            return np.r_[probe.site_xpos[self.edge]-target,
                         0.2*rot[2,0], 0.2*rot[2,1], 0.005*(q-self.neutral)]
        result = least_squares(residual,np.clip(self.neutral,lower,upper),
                               bounds=(lower,upper),max_nfev=200)
        error = np.linalg.norm(residual(result.x)[:3])
        if error > 0.06:
            raise RuntimeError(f"Fixed motor pose unreachable: {error:.3f} mm")
        return result.x

    def reset(self, offset):
        mj.mj_resetDataKeyframe(self.model,self.data,self.key)
        for i in range(self.model.nu):
            j = self.model.actuator_trnid[i,0]
            self.data.ctrl[i] = self.data.qpos[self.model.jnt_qposadr[j]]
        self.onion_xy = self.path_xy+[0,offset]
        self.onion_pos = np.r_[self.onion_xy,self.board_z+0.16]
        for name in ["onion","piece_left","piece_right"]:
            self.model.geom(name).pos[:] = self.onion_pos
            self.model.geom(name).rgba[3] = float(name == "onion")
        self.armed = self.cut = False
        mj.mj_forward(self.model,self.data)
        self.previous = self.data.site_xpos[self.edge].copy()

    def step(self,t,chop):
        phase = min(2.999999,t/self.duration*3)
        if not chop:
            q = self.neutral
        elif phase < 1:
            q = self.neutral+phase*(self.raised-self.neutral)
        elif phase < 2:
            q = self.raised+(phase-1)*(self.lowered-self.raised)
        else:
            q = self.lowered+(phase-2)*(self.raised-self.lowered)
        self.data.ctrl[self.indices] = q
        mj.mj_step(self.model,self.data,nstep=self.substeps)
        mj.mj_forward(self.model,self.data)
        edge = self.data.site_xpos[self.edge].copy()
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Non-finite body state")
        error = np.linalg.norm(edge[:2]-self.onion_xy)
        if edge[2] > self.top_z+0.04 and error < 0.20:
            self.armed = True
        if self.armed and edge[2] <= self.top_z and error > 0.20:
            self.armed = False
        if self.armed and error < 0.20 and edge[2] <= self.board_z+0.055 and edge[2] < self.previous[2]:
            self.cut = True
            self.model.geom("onion").rgba[3] = 0
            for name,direction in [("piece_left",-1),("piece_right",1)]:
                self.model.geom(name).rgba[3] = 1
                self.model.geom(name).pos[:] = self.onion_pos+[0,direction*0.18,0]
        self.previous = edge

    def execute(self, chop):
        for t in np.arange(0,self.duration,self.dt):
            self.step(t,chop)
        return self.cut


if __name__ == "__main__":
    scene = OnionScene()
    for offset,chop in [(0,True),(0,False),(0.6,True)]:
        scene.reset(offset)
        print({"offset":offset,"chop":chop,"cut":scene.execute(chop)},flush=True)
