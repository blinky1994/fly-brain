"""Fixed-target reaching curriculum; the controller remains MotorBrain.

Target coordinates affect reward only, not neural drive. This measures a learned
fixed reach, not visual target tracking. No inverse kinematics or target actions.
"""
import numpy as np
import mujoco as mj
from intact_onion import IntactOnion


class ReachingLesson(IntactOnion):
    tolerance = .025  # model distances are millimetres
    required_dwell = .20

    def __init__(self):
        self.reach_ready = False
        super().__init__()
        start = self.data.site_xpos[self.edge].copy()
        direction = super().observation()[:3]
        self.target = start + .10 * direction / np.linalg.norm(direction)
        self.base_spec.worldbody.add_site(name='reach_target', pos=self.target,
            size=[self.tolerance]*3, type=mj.mjtGeom.mjGEOM_SPHERE,
            rgba=[.1,1,.3,.45])
        self.reach_ready = True
        self.reset()

    def reset(self, offset=0):
        super().reset(offset)
        if not self.reach_ready:
            return
        # Held-out trials perturb the knife's initial posture, NOT the goal.
        self.data.qpos[self.motor_qadr[7]] += offset
        self.data.ctrl[self.motor_indices[7]] += offset
        mj.mj_forward(self.model,self.data)
        self.cutting_enabled = False
        self.start_distance = self.distance()
        self.min_distance = self.start_distance
        self.dwell = self.best_dwell = self.elapsed = self.distance_integral = 0.

    def distance(self):
        return float(np.linalg.norm(self.target-self.data.site_xpos[self.edge]))

    def observation(self):
        obs = super().observation()
        if self.reach_ready:
            obs[:3] = np.clip((self.target-self.data.site_xpos[self.edge])/.5,-1,1)
        return obs

    def step(self, action):
        super().step(action)
        distance = self.distance()
        self.min_distance = min(self.min_distance,distance)
        self.elapsed += self.dt
        self.distance_integral += distance*self.dt
        self.dwell = self.dwell+self.dt if distance <= self.tolerance else 0.
        self.best_dwell = max(self.best_dwell,self.dwell)

    def metrics(self):
        return dict(**super().metrics(), lesson='reach', target_mm=self.target.tolist(),
            start_distance_mm=self.start_distance, final_distance_mm=self.distance(),
            minimum_distance_mm=self.min_distance, tolerance_mm=self.tolerance,
            best_dwell_seconds=self.best_dwell, final_dwell_seconds=self.dwell,
            reach_success=bool(self.dwell+1e-9 >= self.required_dwell))

    def score(self):
        # Reward staying near the target, not a lucky transient crossing.
        mean = self.distance_integral/max(self.elapsed,self.dt)
        return float(1-(.6*self.distance()+.4*mean)/self.start_distance
            + min(self.dwell/self.required_dwell,1)-.02*self.total_effort
            -float(self.hazard>.001))
