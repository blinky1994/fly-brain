"""One rigid onion, split at measured knife planes after a completed incision.

Custom planar cutting approximation, not deformable tissue mechanics. No slices,
seams, or hidden child bodies exist before a cut. The knife remains attached.
"""
import numpy as np
import mujoco as mj
from scipy.spatial import ConvexHull

from onion_physics import PhysicalOnion

RADIUS = np.array([.32, .28, .18])
BASE = -.16
TOTAL_MASS = .0001


def onion_mesh(lo=-.28, hi=.28):
    """Closed convex full onion, with a small flattened base for stability."""
    points = []
    # Include global rings so splitting preserves the original curved surface.
    ys = np.unique(np.r_[lo, np.linspace(-.28, .28, 49)[
        (np.linspace(-.28, .28, 49) > lo) & (np.linspace(-.28, .28, 49) < hi)], hi])
    for y in ys:
        scale = np.sqrt(max(0, 1-(y/RADIUS[1])**2))
        for a in np.linspace(0, 2*np.pi, 64, endpoint=False):
            points.append([RADIUS[0]*scale*np.cos(a), y,
                           max(BASE, RADIUS[2]*scale*np.sin(a))])
    vertices = np.unique(np.round(points, 9), axis=0)
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    center = vertices.mean(axis=0)
    for face in faces:
        a, b, c = vertices[face]
        if np.dot(np.cross(b-a, c-a), (a+b+c)/3-center) < 0:
            face[1], face[2] = face[2], face[1]
    return vertices, faces, hull.volume


def cut_face(spec, body, name, y, side):
    scale = np.sqrt(max(0, 1-(y/.28)**2))
    for band in range(12):
        vertices = []
        for depth in [0, .00008]:
            for r in [max(.001, 1-(band+1)/12), 1-band/12]:
                for a in np.linspace(0, 2*np.pi, 64, endpoint=False):
                    vertices.append([.32*scale*r*np.cos(a), y+side*(.0001+depth),
                                     max(BASE, .18*scale*r*np.sin(a))])
        faces = []
        for i in range(64):
            j = (i+1) % 64
            for a, b in [(0,64), (128,192), (0,128), (64,192)]:
                faces.extend([[a+i,b+i,b+j], [a+i,b+j,a+j]])
        mesh = f'{name}_{band}'
        spec.add_mesh(name=mesh, uservert=np.array(vertices).ravel(), userface=np.array(faces).ravel())
        body.add_geom(name=mesh, type=mj.mjtGeom.mjGEOM_MESH, meshname=mesh, mass=0,
                      contype=0, conaffinity=0,
                      rgba=[.96,.88,.85,1] if band%2 else [.69,.38,.53,1])


class IntactOnion(PhysicalOnion):
    """Direct joint control. No IK, motor profiles, or timed strokes in step()."""
    def __init__(self):
        self.intact_ready = False
        super().__init__()
        spec = self.world.mjcf_root
        spec.delete(next(e for e in spec.equalities if e.name == 'onion_seam'))
        for name in ['onion_left', 'onion_right']:
            spec.delete(next(b for b in spec.bodies if b.name == name))
        for key in list(spec.keys):
            spec.delete(key)
        self.path_xy += [-.15, -.1]
        self.center = np.r_[self.path_xy, self.board_z-BASE+.003]
        for geom in spec.geoms:
            if geom.name == 'blade':
                geom.size = [.34,.004,.16]
                geom.contype, geom.conaffinity = 4, 1
            elif geom.name == 'holding_pad':
                geom.contype, geom.conaffinity = 8, 1
            elif geom.name == 'board':
                geom.friction = [.8,.002,.0001]
        self.neutral_joint = dict(zip(
            [self.model.joint(i).name for i in range(self.model.njnt)
             if self.model.jnt_type[i] != mj.mjtJoint.mjJNT_FREE],
            [self.data.qpos[self.model.jnt_qposadr[i]] for i in range(self.model.njnt)
             if self.model.jnt_type[i] != mj.mjtJoint.mjJNT_FREE]))
        self.serial = 0
        self._add_piece(spec, 'whole_onion', -.28, .28, TOTAL_MASS)
        self.base_spec = spec.copy()
        self.intact_ready = True
        self.revision = 0
        self.reset()

    def _add_piece(self, spec, name, lo, hi, mass):
        vertices, faces, volume = onion_mesh(lo, hi)
        mesh = name+'_shape'
        spec.add_mesh(name=mesh, uservert=vertices.ravel(), userface=faces.ravel())
        body = spec.worldbody.add_body(name=name, pos=self.center)
        body.add_freejoint(name=name+'_free')
        body.add_geom(name=name+'_geom', type=mj.mjtGeom.mjGEOM_MESH, meshname=mesh,
                      mass=mass, rgba=[.48,.12,.25,1], contype=2, conaffinity=1|2|4|8,
                      friction=[.8,.002,.0001], solref=[.0005,1])
        if lo > -.28+1e-6:
            cut_face(spec, body, name+'_front', lo, -1)
        if hi < .28-1e-6:
            cut_face(spec, body, name+'_back', hi, 1)
        return volume

    def _ids(self):
        m = self.model
        self.edge = m.site('edge').id
        self.pad_site = m.site('holding_tip').id
        self.pad_geom = m.geom('holding_pad').id
        self.blade_geom = m.geom('blade').id
        self.motor_indices = np.array([i for i in range(m.nu)
                                      if any(s in m.actuator(i).name for s in ['rf_', 'lf_'])])
        self.motor_joints = m.actuator_trnid[self.motor_indices,0]
        self.motor_qadr = m.jnt_qposadr[self.motor_joints]
        self.motor_dadr = m.jnt_dofadr[self.motor_joints]
        self.limits = m.jnt_range[self.motor_joints].copy()
        self.motor_names = [m.actuator(i).name for i in self.motor_indices]
        self.neutral_motor = np.array([self.neutral_joint[m.joint(j).name] for j in self.motor_joints])
        self.piece_geom = {m.geom(n+'_geom').id:n for n in self.pieces}

    def reset(self, offset=0):
        if not self.intact_ready:
            return super().reset(offset)
        self.spec = self.base_spec.copy()
        self.model = self.spec.compile()
        self.data = mj.MjData(self.model)
        self.pieces = {'whole_onion': dict(lo=-.28, hi=.28, mass=TOTAL_MASS)}
        self._ids()
        for name, value in self.neutral_joint.items():
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = value
        self.data.qpos[self.model.jnt_qposadr[self.model.joint('whole_onion_free').id]+1] += offset
        for i in range(self.model.nu):
            self.data.ctrl[i] = self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[i,0]]]
        mj.mj_forward(self.model, self.data)
        self.initial_center = self.data.xpos[self.model.body('whole_onion').id].copy()
        self.inc = None
        self.events = []
        self.max_slip = self.hazard = self.pad_force = 0.
        self.total_effort = 0.
        self.cutting_enabled = True
        self.dt = .02
        self.previous = self.data.site_xpos[self.edge].copy()
        self.revision += 1

    def remainder(self):
        return max(self.pieces, key=lambda n:self.pieces[n]['mass'])

    def observation(self):
        name = self.remainder()
        body = self.model.body(name).id
        origin = self.data.xpos[body]
        # Goal describes desired outcome, not a commanded trajectory.
        target = origin + [0, self.pieces[name]['lo']+.05, -.14 if self.inc else .22]
        hold = origin + [0,.10,.21]
        return np.r_[np.clip((target-self.data.site_xpos[self.edge])/.5,-1,1),
                     np.clip((hold-self.data.site_xpos[self.pad_site])/.5,-1,1),
                     np.clip(self.data.qpos[self.motor_qadr]-self.neutral_motor,-2,2)/2,
                     float(self.inc is not None), min(self.pad_force/3,1)]

    def _split(self, name, plane):
        """Only called after the incision gate. Preserve pose/velocity; no kick."""
        piece = self.pieces[name]
        if not piece['lo']+.012 < plane < piece['hi']-.012:
            raise ValueError('Cut would create a degenerate piece')
        old_joint = self.model.joint(name+'_free').id
        q = self.data.qpos[self.model.jnt_qposadr[old_joint]:self.model.jnt_qposadr[old_joint]+7].copy()
        v = self.data.qvel[self.model.jnt_dofadr[old_joint]:self.model.jnt_dofadr[old_joint]+6].copy()
        self.spec.delete(next(b for b in self.spec.bodies if b.name == name))
        # Remove obsolete meshes so repeated cutting does not accumulate assets.
        for mesh in list(self.spec.meshes):
            if mesh.name.startswith(name+'_'):
                self.spec.delete(mesh)
        self.serial += 1
        children = [f'cut{self.serial}_left', f'cut{self.serial}_right']
        ranges = [(piece['lo'],plane), (plane,piece['hi'])]
        volumes = [onion_mesh(*r)[2] for r in ranges]
        del self.pieces[name]
        for child, (lo,hi), volume in zip(children,ranges,volumes):
            mass = piece['mass']*volume/sum(volumes)
            self._add_piece(self.spec, child, lo, hi, mass)
            self.pieces[child] = dict(lo=lo,hi=hi,mass=mass)
        self.model, self.data = self.spec.recompile(self.model,self.data)
        self._ids()
        for child in children:
            joint = self.model.joint(child+'_free').id
            self.data.qpos[self.model.jnt_qposadr[joint]:self.model.jnt_qposadr[joint]+7] = q
            self.data.qvel[self.model.jnt_dofadr[joint]:self.model.jnt_dofadr[joint]+6] = v
        mj.mj_forward(self.model,self.data)
        self.revision += 1

    def _cutting(self):
        edge = self.data.site_xpos[self.edge].copy()
        downward = edge[2] < self.previous[2]-1e-9
        contacts = {}
        pad = hazard = 0.
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {int(c.geom1),int(c.geom2)}
            if self.blade_geom not in pair and self.pad_geom not in pair:
                continue
            force = np.zeros(6)
            mj.mj_contactForce(self.model,self.data,i,force)
            normal = max(0,float(force[0]))
            if pair == {self.blade_geom,self.pad_geom}:
                hazard += normal
            for geom in pair & self.piece_geom.keys():
                if self.blade_geom in pair:
                    contacts[self.piece_geom[geom]] = normal
                if self.pad_geom in pair:
                    pad += normal
        self.pad_force = pad
        self.hazard = max(self.hazard,hazard)
        remainder = self.model.body(self.remainder()).id
        self.max_slip = max(self.max_slip,float(np.linalg.norm(
            self.data.xpos[remainder,:2]-self.initial_center[:2])))
        if self.inc is None and self.cutting_enabled and len(self.events)<8 and downward:
            for name, force in contacts.items():
                body = self.model.body(name).id
                rot = self.data.xmat[body].reshape(3,3)
                local = rot.T @ (edge-self.data.xpos[body])
                normal = self.data.site_xmat[self.edge].reshape(3,3)[:,1]
                tilt = np.arccos(np.clip(abs(normal@rot[:,1]),0,1))
                p = self.pieces[name]
                if (force>.001 and tilt<np.deg2rad(10) and local[2]>.02
                        and p['lo']+.012 < local[1] < p['hi']-.012 and abs(local[0])<.05):
                    self.inc = dict(name=name,plane=float(local[1]),start_z=float(local[2]),
                                    depth=0.,work=0.,valid=True,held=pad>.001)
                    # Explicit custom penetration replaces the rigid blade contact
                    # for this body only. Board/holding/piece contacts stay enabled.
                    self.model.geom(name+'_geom').conaffinity = 1|2|8
                    break
        if self.inc is not None:
            inc = self.inc
            body = self.model.body(inc['name']).id
            rot = self.data.xmat[body].reshape(3,3)
            local = rot.T @ (edge-self.data.xpos[body])
            knife = self.data.site_xmat[self.edge].reshape(3,3)
            tilt = np.arccos(np.clip(abs(knife[:,1]@rot[:,1]),0,1))
            inc['valid'] &= bool(abs(local[1]-inc['plane'])<.025 and abs(local[0])<.05 and tilt<np.deg2rad(10))
            inc['held'] &= pad>.001
            if downward:
                dz = max(0,float(self.previous[2]-edge[2]))
                inc['work'] += 1.5*dz
                inc['depth'] = max(inc['depth'],inc['start_z']-float(local[2]))
            highest = local[2]+.34*abs((rot.T@knife[:,0])[2])
            # Near an end, the curved onion's underside is ABOVE its central
            # flat base. Complete the local cross-section, not empty space below.
            bottom = max(BASE,-RADIUS[2]*np.sqrt(1-(inc['plane']/RADIUS[1])**2))
            if downward and inc['valid'] and highest<=bottom+.003 and inc['depth']>.10 and inc['work']>.10:
                width = inc['plane']-self.pieces[inc['name']]['lo']
                clean = bool(inc['held'] and .025<=width<=.08 and self.max_slip<.08 and self.hazard<=.001)
                self.events.append(dict(plane_y_mm=inc['plane'],thickness_mm=width,
                                        clean=clean,held=bool(inc['held']),work=inc['work']))
                self._split(inc['name'],inc['plane'])
                self.inc = None
            elif local[2] > .22:
                self.model.geom(inc['name']+'_geom').conaffinity = 1|2|4|8
                self.inc = None
        self.previous = edge

    def step(self, action):
        action = np.asarray(action,dtype=float)
        if action.shape != (len(self.motor_indices),) or not np.isfinite(action).all():
            raise ValueError('Expected one finite velocity command per front-leg joint')
        velocity = np.clip(action,-1,1)*1.5
        self.data.ctrl[self.motor_indices] = np.clip(
            self.data.ctrl[self.motor_indices]+velocity*self.dt,
            self.limits[:,0]+.002,self.limits[:,1]-.002)
        self.total_effort += float(np.mean(velocity**2))*self.dt
        for _ in range(round(self.dt/self.model.opt.timestep)):
            self.data.qfrc_applied[:] = 0
            if self.inc:
                point = self.data.site_xpos[self.edge].copy()
                mj.mj_applyFT(self.model,self.data,[0,0,1.5],[0,0,0],point,
                             int(self.model.geom_bodyid[self.blade_geom]),self.data.qfrc_applied)
                mj.mj_applyFT(self.model,self.data,[0,0,-1.5],[0,0,0],point,
                             self.model.body(self.inc['name']).id,self.data.qfrc_applied)
            mj.mj_step(self.model,self.data)
            self._cutting()
        if not np.isfinite(self.data.qpos).all() or np.max(abs(self.data.qvel))>1e6:
            raise RuntimeError('Unstable physical trial')

    def metrics(self):
        return dict(physical_pieces=len(self.pieces),cuts=len(self.events),
                    clean_slices=sum(e['clean'] for e in self.events),events=self.events,
                    max_slip_mm=self.max_slip,blade_pad_force=self.hazard,
                    intact=len(self.pieces)==1,incision_active=self.inc is not None)

    def score(self):
        obs = self.observation()
        # Dense reach/hold shaping supports early learning; only events score cuts.
        reach = np.exp(-4*np.linalg.norm(obs[:3]))
        hold = np.exp(-4*np.linalg.norm(obs[3:6]))
        return float(.25*reach+.15*hold+sum(1+2*e['clean'] for e in self.events)
                     -.1*min(self.max_slip/.08,2)-.05*self.total_effort
                     -float(self.hazard>.001))
