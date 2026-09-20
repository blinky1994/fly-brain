"""Supported fly pressing a passive, vertically guided bar with assisted grips.

Only front-leg joints are actuated. The bar has no actuator or scripted motion.
Guides keep it level and passive constraints secure the front feet to the bar.
This first lesson tests lifting, not balance or learning to grasp.
"""
import os
from pathlib import Path
import mujoco as mj
import numpy as np
from flygym.compose import ActuatorType, KinematicPosePreset, TetheredWorld
from flygym.compose.fly import FlyBody
from flygym.flybody import FlyBodyActuatedDOFPreset, FlyBodyAxisOrder
from flygym.flybody import FlyBodyJointPreset, FlyBodySkeleton
from flygym.utils.math import Rotation3D


class BenchPress:
    target_height = .15
    required_hold = .20

    def __init__(self):
        assets = Path(__file__).resolve().parent/'data'
        os.environ.setdefault('FLYGYM_ASSET_CACHE_DIR',str(assets))
        fly = FlyBody(name='lifter')
        skeleton = FlyBodySkeleton(axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
                                   joint_preset=FlyBodyJointPreset.LEGS_ONLY)
        fly.add_joints(skeleton,KinematicPosePreset.FLYBODY_NEUTRAL)
        fly.add_actuators(skeleton.get_actuated_dofs_from_preset(
            FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY),ActuatorType.POSITION,kp=100)
        fly.colorize()
        world = TetheredWorld()
        world.add_fly(fly,[0,0,3.0],Rotation3D('quat',[0,1,0,0]))
        spec = world.mjcf_root
        for geom in spec.geoms: geom.contype = geom.conaffinity = 0
        model,data = world.compile()
        mj.mj_resetDataKeyframe(model,data,model.key('neutral').id)
        mj.mj_forward(model,data)
        neutral = {model.joint(i).name:float(data.qpos[model.jnt_qposadr[i]])
                   for i in range(model.njnt) if model.jnt_type[i]!=mj.mjtJoint.mjJNT_FREE}
        feet,tip_offsets = [],{}
        for side in ['lf','rf']:
            # The tarsus1 origin is the proximal foot joint, not the claw tip.
            # Locate the distal surface of the actual final foot mesh.
            body = model.body(f'lifter/{side}_tarsus5').id
            proximal = model.body(f'lifter/{side}_tarsus4').id
            direction = data.xpos[body]-data.xpos[proximal]
            direction /= np.linalg.norm(direction)
            geom = next(i for i in range(model.ngeom) if model.geom_bodyid[i]==body
                        and model.geom_type[i]==mj.mjtGeom.mjGEOM_MESH)
            mesh = model.geom_dataid[geom]
            start,count = model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3,3).T + data.geom_xpos[geom]
            projection = vertices @ direction
            tip = vertices[projection>=np.quantile(projection,.95)].mean(axis=0)
            tip_offsets[side] = data.xmat[body].reshape(3,3).T @ (tip-data.xpos[body])
            feet.append(tip)
        feet = np.asarray(feet)
        self.bar_origin = feet.mean(axis=0)
        self.bar_origin[2] = feet[:,2].mean()
        self.half_width = max(.5,float(np.ptp(feet[:,1])/2+.2))
        for side in ['lf','rf']:
            foot = next(b for b in spec.bodies if b.name==f'lifter/{side}_tarsus5')
            foot.add_geom(name=f'{side}_press_pad',type=mj.mjtGeom.mjGEOM_SPHERE,
                pos=tip_offsets[side],size=[.055,0,0],mass=1e-7,rgba=[.95,.6,.15,1],
                contype=0,conaffinity=0,friction=[1,.005,.0001])
            foot.add_site(name=f'{side}_tip',pos=tip_offsets[side],size=[.015]*3,rgba=[1,1,1,1])
        for key in list(spec.keys): spec.delete(key)
        wb = spec.worldbody
        wb.add_geom(name='floor',type=mj.mjtGeom.mjGEOM_PLANE,size=[6,6,.1],
                    rgba=[.10,.14,.20,1],contype=0,conaffinity=0)
        wb.add_geom(name='bench',type=mj.mjtGeom.mjGEOM_BOX,pos=[0,0,1.25],
                    size=[1.65,.48,.12],rgba=[.16,.30,.55,1],contype=0,conaffinity=0)
        for x in [-1.2,1.2]:
            wb.add_geom(type=mj.mjtGeom.mjGEOM_BOX,pos=[x,0,.6],size=[.08,.38,.6],
                        rgba=[.32,.36,.42,1],contype=0,conaffinity=0)
        for y in [-self.half_width,self.half_width]:
            x,cy,z = self.bar_origin
            wb.add_geom(type=mj.mjtGeom.mjGEOM_CAPSULE,
                fromto=[x,cy+y,.1,x,cy+y,z+.65],size=[.025,0,0],
                rgba=[.35,.4,.46,1],contype=0,conaffinity=0)
        wb.add_site(name='lift_goal',type=mj.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*(self.bar_origin+[0,-self.half_width,self.target_height]),
                    *(self.bar_origin+[0,self.half_width,self.target_height])],
            size=[.012,0,0],rgba=[.2,1,.5,.5])
        bar = wb.add_body(name='barbell',pos=self.bar_origin)
        bar.add_joint(name='bar_slide',type=mj.mjtJoint.mjJNT_SLIDE,axis=[0,0,1],
                      limited=True,range=[0,.5],damping=.002)
        for side,position in zip(['lf','rf'],feet):
            bar.add_site(name=f'{side}_bar_grip',pos=position-self.bar_origin,
                         size=[.012]*3,rgba=[.2,1,.5,1])
            spec.add_equality(name=f'{side}_grip',type=mj.mjtEq.mjEQ_CONNECT,
                objtype=mj.mjtObj.mjOBJ_SITE,name1=f'{side}_tip',name2=f'{side}_bar_grip',
                solref=[.002,1],solimp=[.99,.999,.001,.5,2])
        bar.add_geom(name='bar_shaft',type=mj.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0,-self.half_width,0,0,self.half_width,0],size=[.035,0,0],
            mass=1e-5,rgba=[.8,.85,.9,1],contype=2,conaffinity=1,
            solref=[.002,1],friction=[1,.005,.0001])
        # Six gym-style plates. "20 KG" is a visual label: preserve the original
        # total simulated plate mass so existing training remains comparable.
        glyphs = {'2':['111','001','111','100','111'],
                  '0':['111','101','101','101','111'],
                  'K':['101','110','100','110','101'],
                  'G':['111','100','101','101','111']}
        for side in [-1,1]:
            bar.add_geom(type=mj.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0,side*self.half_width,0,0,side*(self.half_width+.43),0],
                size=[.045,0,0],mass=0,rgba=[.65,.7,.76,1],contype=0,conaffinity=0)
            for plate in range(3):
                y=side*(self.half_width+.075+plate*.11)
                bar.add_geom(name=f'plate_{side}_{plate}',type=mj.mjtGeom.mjGEOM_CYLINDER,
                    pos=[0,y,0],quat=[.70710678,.70710678,0,0],size=[.34,.047,0],
                    mass=2e-6/3,rgba=[.10,.30+.025*plate,.68,1],contype=0,conaffinity=0)
                bar.add_geom(type=mj.mjtGeom.mjGEOM_CYLINDER,pos=[0,y+side*.048,0],
                    quat=[.70710678,.70710678,0,0],size=[.068,.003,0],mass=0,
                    rgba=[.72,.77,.83,1],contype=0,conaffinity=0)
                for word,z in [('20',.16),('KG',-.16)]:
                    for letter,char in enumerate(word):
                        for row,pixels in enumerate(glyphs[char]):
                            for col,pixel in enumerate(pixels):
                                if pixel=='1':
                                    bar.add_geom(type=mj.mjtGeom.mjGEOM_BOX,
                                        pos=[-side*((letter*4+col-3)*.026),y+side*.048,z+(2-row)*.026],
                                        size=[.0105,.002,.0105],mass=0,rgba=[.95,.97,1,1],
                                        contype=0,conaffinity=0)
        self.model = spec.compile()
        self.model.opt.timestep = .0001
        self.data = mj.MjData(self.model)
        self.neutral = neutral
        m = self.model
        self.motor_indices = np.array([i for i in range(m.nu)
            if any(s in m.actuator(i).name for s in ['lf_','rf_'])])
        joints = m.actuator_trnid[self.motor_indices,0]
        self.motor_qadr = m.jnt_qposadr[joints]
        self.limits = m.jnt_range[joints].copy()
        self.motor_names = [m.actuator(i).name for i in self.motor_indices]
        self.neutral_motor = np.array([neutral[m.joint(j).name] for j in joints])
        self.bar_qadr = m.jnt_qposadr[m.joint('bar_slide').id]
        self.bar_dadr = m.jnt_dofadr[m.joint('bar_slide').id]
        self.pad_ids = [m.geom(f'{s}_press_pad').id for s in ['lf','rf']]
        self.bar_geom = m.geom('bar_shaft').id
        self.grip_ids = [m.equality(f'{s}_grip').id for s in ['lf','rf']]
        self.dt,self.revision = .02,0
        self.camera_target = self.bar_origin+[0,0,-.35]
        self.reset()

    def reset(self,offset=0):
        mj.mj_resetData(self.model,self.data)
        for name,value in self.neutral.items():
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = value
        self.data.qpos[self.motor_qadr[[0,7]]] += offset
        for i in range(self.model.nu):
            self.data.ctrl[i] = self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[i,0]]]
        mj.mj_forward(self.model,self.data)
        self.max_height = self.integral_height = self.effort = self.elapsed = self.hold = 0.
        self.pad_force = np.zeros(2)
        self.bilateral_time = 0.

    def height(self):
        return max(0.,float(self.data.qpos[self.bar_qadr]))

    def observation(self):
        # Keep the 22-value motor interface. First six task errors are not
        # injected into the brain; existing proprioceptors encode joint angles.
        return np.r_[np.zeros(6),
            np.clip(self.data.qpos[self.motor_qadr]-self.neutral_motor,-2,2)/2,
            0.,min(float(self.pad_force.mean())/3,1)]

    def step(self,action):
        action = np.asarray(action,dtype=float)
        if action.shape!=(14,) or not np.isfinite(action).all():
            raise ValueError('Expected 14 finite front-leg commands')
        velocity = np.clip(action,-1,1)*1.5
        self.data.ctrl[self.motor_indices] = np.clip(
            self.data.ctrl[self.motor_indices]+velocity*self.dt,
            self.limits[:,0]+.002,self.limits[:,1]-.002)
        self.effort += float(np.mean(velocity**2))*self.dt
        force = np.zeros(2)
        substeps = round(self.dt/self.model.opt.timestep)
        for _ in range(substeps):
            mj.mj_step(self.model,self.data)
            for side,equality in enumerate(self.grip_ids):
                rows=(self.data.efc_type==mj.mjtConstraint.mjCNSTR_EQUALITY) & (self.data.efc_id==equality)
                force[side] += float(np.linalg.norm(self.data.efc_force[rows]))/substeps
        if not np.isfinite(self.data.qpos).all() or np.max(abs(self.data.qvel))>1e6:
            raise RuntimeError('Unstable bench physics')
        self.pad_force = force
        h = self.height()
        self.elapsed += self.dt
        self.max_height = max(h,self.max_height)
        self.integral_height += min(h/self.target_height,1)*self.dt
        both = bool(np.all(self.data.eq_active[self.grip_ids]) and np.all(self.grip_errors()<.02))
        self.bilateral_time += self.dt*both
        self.hold = self.hold+self.dt if h>=self.target_height and both and abs(self.data.qvel[self.bar_dadr])<.2 else 0.

    def grip_errors(self):
        return np.array([np.linalg.norm(self.data.site_xpos[self.model.site(f'{s}_tip').id]-
                        self.data.site_xpos[self.model.site(f'{s}_bar_grip').id]) for s in ['lf','rf']])

    def metrics(self):
        return dict(lesson='bench',height_mm=self.height(),max_height_mm=self.max_height,
            target_height_mm=self.target_height,hold_seconds=self.hold,
            left_grip_force=float(self.pad_force[0]),right_grip_force=float(self.pad_force[1]),
            grip_errors_mm=self.grip_errors().tolist(),assisted_grip=True,
            bilateral_grip_seconds=self.bilateral_time,
            lift_success=bool(self.hold+1e-9>=self.required_hold),bar_guided=True)

    def score(self):
        return float(.6*min(self.height()/self.target_height,1)
            +.4*self.integral_height/max(self.elapsed,self.dt)
            +min(self.hold/self.required_hold,1)-.02*self.effort)

    def status(self):
        return (f'Bar lift: {self.height():.3f} / {self.target_height:.3f} mm | '
                f'hold: {self.hold:.2f} / {self.required_hold:.2f} s\n'
                'ASSISTED GRIP: feet secured to bar | learn to lift, no full reps yet')
