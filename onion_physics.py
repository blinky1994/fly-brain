"""Two-front-leg physical prototype. Contact/friction hold; predefined breakable seam.

The torso is supported. Both leg trajectories are engineered, not newly trained.
Onion halves have free joints; only their mutual seam is constrained before cutting.
"""
import argparse
import json
import time
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from onion_scene import OnionScene

ROOT = Path(__file__).resolve().parent


class PhysicalOnion(OnionScene):
    def __init__(self):
        self.physical_ready = False
        super().__init__()
        spec = self.world.mjcf_root
        shift = -self.path_xy[1]
        self.path_xy[1] = 0
        self.radius = np.array([0.32,0.28,0.18])
        self.top_z = self.board_z+2*self.radius[2]
        for geom in list(spec.geoms):
            if geom.name in ["onion","piece_left","piece_right"]:
                spec.delete(geom)
            elif geom.name in ["board","table","leg0","leg1","leg2","leg3"]:
                geom.pos[1] += shift
            if geom.name == "board":
                geom.friction = [0.35,0.001,0.0001]
            if geom.contype:
                geom.solref = [0.0005,1]
        foot = next(b for b in spec.bodies if b.name == "chef/lf_tarsus1")
        foot.add_geom(name="holding_pad",type=mj.mjtGeom.mjGEOM_SPHERE,size=[0.07]*3,
            mass=1e-8,rgba=[0.45,0.25,0.08,1],contype=1,conaffinity=1,
            friction=[1.2,0.005,0.001],solref=[0.0005,1])
        foot.add_site(name="holding_tip",pos=[0,0,0],size=[0.015]*3,rgba=[0,1,0,0])
        # Ellipsoid halves are convex meshes with an actual flat cut face.
        for name,side in [("onion_left",1),("onion_right",-1)]:
            verts = []
            for theta in np.linspace(0,np.pi/2,10):
                for phi in np.linspace(0,2*np.pi,32,endpoint=False):
                    verts.append([self.radius[0]*np.sin(theta)*np.cos(phi),
                                  side*self.radius[1]*np.cos(theta),
                                  self.radius[2]*np.sin(theta)*np.sin(phi)])
            verts = np.unique(np.round(verts,8),axis=0)
            faces = ConvexHull(verts).simplices.copy()
            center = verts.mean(axis=0)
            for face in faces:
                a,b,c = verts[face]
                if np.dot(np.cross(b-a,c-a),(a+b+c)/3-center) < 0:
                    face[1],face[2] = face[2],face[1]
            spec.add_mesh(name=name+"_mesh",uservert=verts.ravel(),userface=faces.ravel())
            body = spec.worldbody.add_body(name=name,
                pos=[*self.path_xy,self.board_z+self.radius[2]+0.005])
            joint = body.add_freejoint(name=name+"_free")
            self.world.world_dof_neutral_states.add(joint.name)
            body.add_geom(name=name+"_geom",type=mj.mjtGeom.mjGEOM_MESH,
                meshname=name+"_mesh",mass=0.00005,rgba=[0.65,0.28,0.5,1],
                contype=1,conaffinity=1,friction=[0.6,0.002,0.0001],
                solref=[0.0005,1])
        spec.add_equality(name="onion_seam",type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_BODY,
            name1="onion_left",name2="onion_right",solref=[0.0005,1])
        self.world._neutral_keyframe.qpos = []
        self.world._rebuild_neutral_keyframe()
        self.model,self.data = self.world.compile()
        self.key = self.model.key("neutral").id
        self.edge = self.model.site("edge").id
        self.pad_site = self.model.site("holding_tip").id
        self.pad_geom = self.model.geom("holding_pad").id
        self.blade_geom = self.model.geom("blade").id
        self.piece_geoms = {self.model.geom(n+"_geom").id for n in ["onion_left","onion_right"]}
        self.piece_bodies = [self.model.body(n).id for n in ["onion_left","onion_right"]]
        self.piece_qadr = [self.model.jnt_qposadr[self.model.joint(n+"_free").id]
                           for n in ["onion_left","onion_right"]]
        self.seam = self.model.equality("onion_seam").id
        self.left_indices = np.array([i for i in range(self.model.nu) if "lf_" in self.model.actuator(i).name])
        self.left_joints = self.model.actuator_trnid[self.left_indices,0]
        self.left_qadr = self.model.jnt_qposadr[self.left_joints]
        mj.mj_resetDataKeyframe(self.model,self.data,self.key)
        self.left_neutral = self.data.qpos[self.left_qadr].copy()
        self.neutral = self.data.qpos[self.qadr].copy()
        self.raised = self.solve_pose(np.r_[self.path_xy,self.top_z+0.22])
        self.lowered = self.solve_pose(np.r_[self.path_xy,self.board_z+0.015])
        self.hold_target = np.r_[self.path_xy+[0,0.14],self.board_z+0.18+0.156+0.07-0.02]
        self.left_hold = self.solve_left(self.hold_target)
        self.duration = 2.4
        self.fracture_work = 0.02  # Chosen simulation-unit parameter, not onion material data.
        self.physical_ready = True
        self.reset(0)

    def solve_left(self,target):
        probe = mj.MjData(self.model)
        mj.mj_resetDataKeyframe(self.model,probe,self.key)
        lo = self.model.jnt_range[self.left_joints,0]+0.001
        hi = self.model.jnt_range[self.left_joints,1]-0.001
        def residual(q):
            probe.qpos[self.left_qadr] = q
            mj.mj_forward(self.model,probe)
            return np.r_[probe.site_xpos[self.pad_site]-target,0.003*(q-self.left_neutral)]
        r = least_squares(residual,np.clip(self.left_neutral,lo,hi),bounds=(lo,hi),max_nfev=300)
        error = np.linalg.norm(residual(r.x)[:3])
        if error > 0.06:
            raise RuntimeError(f"Holding foot cannot reach onion: {error:.3f} mm")
        return r.x

    def reset(self,offset=0):
        if not self.physical_ready:
            return super().reset(offset)
        mj.mj_resetDataKeyframe(self.model,self.data,self.key)
        self.data.eq_active[self.seam] = True
        for i in range(self.model.nu):
            j = self.model.actuator_trnid[i,0]
            self.data.ctrl[i] = self.data.qpos[self.model.jnt_qposadr[j]]
        self.onion_xy = self.path_xy+[0,offset]
        if abs(offset) <= 0.12:
            self.left_hold = self.solve_left(self.hold_target+[0,offset,0])
        for adr in self.piece_qadr:
            self.data.qpos[adr+1] += offset
        mj.mj_forward(self.model,self.data)
        self.initial_centers = self.data.xpos[self.piece_bodies].copy()
        self.previous = self.data.site_xpos[self.edge].copy()
        self.cut = False
        self.work = self.hold_time = self.slip = self.knife_pad_force = 0.0
        self.peak_blade_force = self.peak_pad_force = 0.0
        self.current_pad_force = 0.0
        self.first_pad_contact = None
        self.pad_force_at_fracture = None
        self.disturbance_force = 0.0
        self.hold_enabled = abs(offset) <= 0.12
        self.fracture_enabled = True
        self.cut_time = None

    def step(self,t,chop):
        approach = np.clip(t/0.6,0,1)
        left = self.left_neutral+approach*(self.left_hold-self.left_neutral) if self.hold_enabled else self.left_neutral
        if not chop or t < 0.9:
            right = self.neutral+approach*(self.raised-self.neutral)
        elif t < 1.5:
            right = self.raised+((t-0.9)/0.6)*(self.lowered-self.raised)
        else:
            right = self.lowered+min(1,(t-1.5)/0.3)*(self.raised-self.lowered)
        self.data.ctrl[self.indices] = right
        self.data.ctrl[self.left_indices] = left
        for body in self.piece_bodies:
            self.data.xfrc_applied[body,:] = 0
            if 0.85 <= t < 0.90:
                self.data.xfrc_applied[body,0] = self.disturbance_force/2
        for _ in range(self.substeps):
            mj.mj_step(self.model,self.data)
            blade_force = pad_force = hazard = 0.0
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                pair = {int(c.geom1),int(c.geom2)}
                if (self.blade_geom in pair or self.pad_geom in pair):
                    force = np.zeros(6)
                    mj.mj_contactForce(self.model,self.data,i,force)
                    normal = max(0,float(force[0]))
                    if pair & self.piece_geoms:
                        if self.blade_geom in pair:
                            blade_force += normal
                        if self.pad_geom in pair:
                            pad_force += normal
                    if pair == {self.blade_geom,self.pad_geom}:
                        hazard += normal
            edge = self.data.site_xpos[self.edge].copy()
            dz = max(0,float(self.previous[2]-edge[2]))
            self.work += blade_force*dz
            self.peak_blade_force = max(self.peak_blade_force,blade_force)
            self.peak_pad_force = max(self.peak_pad_force,pad_force)
            self.current_pad_force = pad_force
            if pad_force > 0.001 and self.first_pad_contact is None:
                self.first_pad_contact = float(self.data.time)
            self.hold_time += (pad_force > 0.001)*self.model.opt.timestep
            self.knife_pad_force = max(self.knife_pad_force,hazard)
            if not self.cut:
                drift = np.linalg.norm(self.data.xpos[self.piece_bodies,:2]-self.initial_centers[:,:2],axis=1).max()
                self.slip = max(self.slip,float(drift))
            # Contact work, downward motion, and actual contact are required.
            # No visual replacement or teleporting occurs at fracture.
            if self.fracture_enabled and not self.cut and self.work >= self.fracture_work and blade_force > 0.01 and dz > 0:
                self.data.eq_active[self.seam] = False
                self.cut = True
                self.cut_time = float(self.data.time)
                self.pad_force_at_fracture = pad_force
            self.previous = edge
        mj.mj_forward(self.model,self.data)
        if not np.isfinite(self.data.qpos).all() or np.max(np.abs(self.data.qvel)) > 1e6:
            raise RuntimeError("Unstable physical trial")

    def metrics(self):
        return {"seam_broken":bool(self.cut),"cut_time_s":self.cut_time,
            "first_pad_contact_s":self.first_pad_contact,
            "pad_force_at_fracture_model_units":self.pad_force_at_fracture,
            "pad_contact_s":float(self.hold_time),"peak_pad_force_model_units":float(self.peak_pad_force),
            "peak_blade_force_model_units":float(self.peak_blade_force),
            "prefracture_slip_mm":float(self.slip),"knife_pad_force_model_units":float(self.knife_pad_force),
            "contact_work_model_units":float(self.work),
            "relative_body_origin_separation_mm":float(np.linalg.norm(self.data.xpos[self.piece_bodies[0]]-self.data.xpos[self.piece_bodies[1]])),
            "piece_COM_separation_mm":float(np.linalg.norm(self.data.xipos[self.piece_bodies[0]]-self.data.xipos[self.piece_bodies[1]]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["check","watch"])
    args = p.parse_args()
    scene = PhysicalOnion()
    if args.mode == "check":
        results = {}
        for name,hold,chop,fracture,nudge in [("hold_and_cut",True,True,True,0),("no_hold",False,True,True,0),
                                       ("hold_only",True,False,True,0),("unbreakable",True,True,False,0),
                                       ("nudged_hold",True,True,True,0.15),("nudged_no_hold",False,True,True,0.15)]:
            scene.reset()
            scene.hold_enabled,scene.fracture_enabled = hold,fracture
            scene.disturbance_force = nudge
            scene.execute(chop)
            results[name] = scene.metrics()
            print(name,json.dumps(results[name]),flush=True)
        out = ROOT/"data/experiments/physical_onion"
        out.mkdir(parents=True,exist_ok=True)
        (out/"report.json").write_text(json.dumps(results,indent=2))
        held = results["hold_and_cut"]
        assert held["seam_broken"] and held["first_pad_contact_s"] < held["cut_time_s"]
        assert held["pad_force_at_fracture_model_units"] > 0.001
        assert not results["hold_only"]["seam_broken"] and not results["unbreakable"]["seam_broken"]
        assert all(r["knife_pad_force_model_units"] == 0 for r in results.values())
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(scene.model,scene.data) as v:
            v.cam.lookat[:] = [*scene.path_xy,scene.board_z+0.3]
            v.cam.distance,v.cam.azimuth,v.cam.elevation = 6.5,110,-30
            while v.is_running():
                with v.lock():
                    scene.reset()
                for t in np.arange(0,scene.duration,scene.dt):
                    if not v.is_running():
                        return
                    started = time.perf_counter()
                    with v.lock():
                        scene.step(t,True)
                    v.set_texts((None,None,"Physical onion prototype\nLeft foot contact\nSeam\nControl",
                        f"\n{'yes' if scene.current_pad_force > 0.001 else 'no'}\n{'broken' if scene.cut else 'intact'}\nFixed two-leg motion; not neural training"))
                    v.sync()
                    time.sleep(max(0,scene.dt*2-(time.perf_counter()-started)))
                deadline = time.perf_counter()+1
                while v.is_running() and time.perf_counter() < deadline:
                    v.sync()
                    time.sleep(0.02)


if __name__ == "__main__":
    main()
