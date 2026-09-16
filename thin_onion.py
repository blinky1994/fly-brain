"""Layered miniature onion with physical slabs and contact-triggered seams.

Rigid predefined fracture surfaces, not deformable onion tissue. Distances are
in the existing FlyGym millimetre-scale scene, not kitchen-scale measurements.
"""
import numpy as np
import mujoco as mj
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares

from onion_physics import PhysicalOnion


def smooth(value):
    u = np.clip(value, 0, 1)
    return u*u*(3-2*u)


def slab_mesh(radius, lo, hi):
    points = []
    for y in np.linspace(lo, hi, 9):
        scale = np.sqrt(max(0, 1-(y/radius[1])**2))
        for a in np.linspace(0, np.pi, 64):
            points.append([radius[0]*scale*np.cos(a), y, radius[2]*scale*np.sin(a)])
    vertices = np.unique(np.round(points, 9), axis=0)
    faces = ConvexHull(vertices).simplices.copy()
    center = vertices.mean(axis=0)
    for face in faces:
        a, b, c = vertices[face]
        if np.dot(np.cross(b-a,c-a), (a+b+c)/3-center) < 0:
            face[1], face[2] = face[2], face[1]
    return vertices, faces


def add_face(spec, body, name, radius, y, side):
    """Thin noncolliding annuli on each cut face, attached to its physical piece."""
    cross = np.sqrt(max(0, 1-(y/radius[1])**2))
    if cross < 0.02:
        return
    # Closed, very thin annular meshes avoid coplanar-mesh compiler problems.
    for band in range(14):
        outer = 1-band/14
        inner = max(0.001, outer-1/14)
        vertices = []
        for depth in [0, 0.00012]:
            for r in [inner,outer]:
                for a in np.linspace(0,np.pi,64):
                    wobble = 1-0.006*(1+np.sin(5*a+band*0.4))
                    vertices.append([radius[0]*cross*r*np.cos(a)*wobble,
                                     y+side*(0.00015+depth),radius[2]*cross*r*np.sin(a)*wobble])
        faces = []
        for i in range(64):
            j = (i+1)%64
            for ring1,ring2 in [(0,64),(128,192),(0,128),(64,192)]:
                faces.extend([[ring1+i,ring2+i,ring2+j],[ring1+i,ring2+j,ring1+j]])
        mesh = f"{name}_layer{band}"
        spec.add_mesh(name=mesh,uservert=np.array(vertices).ravel(),userface=np.array(faces).ravel())
        color = [0.87,0.64,0.75,1] if band%2 == 0 else [0.97,0.92,0.85,1]
        body.add_geom(name=mesh,type=mj.mjtGeom.mjGEOM_MESH,meshname=mesh,
                      rgba=color,contype=0,conaffinity=0,mass=0)


class ThinOnion(PhysicalOnion):
    def __init__(self):
        self.thin_ready = False
        super().__init__()
        spec = self.world.mjcf_root
        spec.delete(next(e for e in spec.equalities if e.name == "onion_seam"))
        for name in ["onion_left","onion_right"]:
            self.world.world_dof_neutral_states.discard(name+"_free")
            spec.delete(next(b for b in spec.bodies if b.name == name))
        # Five strokes produce five thin end slices and leave a grippable heel.
        self.boundaries = np.array([-0.28,-0.21,-0.14,-0.07,0,0.07,0.28])
        self.slice_width = 0.07
        self.cut_planes = self.boundaries[1:-1]
        # A prepared half onion rests on its flat face, as for controlled slicing.
        self.radius[2]=.36
        self.top_z=self.board_z+.36
        self.piece_names = [f"slice{i}" for i in range(len(self.boundaries)-1)]
        for i,(lo,hi) in enumerate(zip(self.boundaries[:-1],self.boundaries[1:])):
            vertices,faces = slab_mesh(self.radius,lo,hi)
            name = self.piece_names[i]
            spec.add_mesh(name=name+"_mesh",uservert=vertices.ravel(),userface=faces.ravel())
            body = spec.worldbody.add_body(name=name,pos=[*self.path_xy,self.board_z+0.003])
            joint = body.add_freejoint(name=name+"_free")
            self.world.world_dof_neutral_states.add(joint.name)
            integral = lambda y: y-y**3/(3*self.radius[1]**2)
            mass = 0.0001*(integral(hi)-integral(lo))/(4*self.radius[1]/3)
            body.add_geom(name=name+"_geom",type=mj.mjtGeom.mjGEOM_MESH,meshname=name+"_mesh",
                          mass=mass,rgba=[0.48,0.12,0.25,1],contype=1,conaffinity=1,
                          friction=[0.7,0.002,0.0001],solref=[0.0005,1])
            add_face(spec,body,name+"_front",self.radius,lo,-1)
            add_face(spec,body,name+"_back",self.radius,hi,1)
            if i:
                spec.add_equality(name=f"slice_seam{i-1}",type=mj.mjtEq.mjEQ_WELD,
                                  objtype=mj.mjtObj.mjOBJ_BODY,name1=self.piece_names[i-1],
                                  name2=name,solref=[0.0005,1])
        # Longitudinal skin striations on the ellipsoid exterior. No contact mass.
        for i,(lo,hi) in enumerate(zip(self.boundaries[:-1],self.boundaries[1:])):
            body = next(b for b in spec.bodies if b.name == self.piece_names[i])
            for j,a in enumerate(np.linspace(0,np.pi,28)):
                ys = np.linspace(max(lo,-.279),min(hi,.279),8)
                for k,(ya,yb) in enumerate(zip(ys[:-1],ys[1:])):
                    pts=[]
                    for y in [ya,yb]:
                        scale=np.sqrt(max(0,1-(y/.28)**2))
                        pts.extend([.3203*scale*np.cos(a),y,.3603*scale*np.sin(a)])
                    body.add_geom(name=f"skin_{i}_{j}_{k}",type=mj.mjtGeom.mjGEOM_CAPSULE,
                                  fromto=pts,size=[.0007]*3,rgba=[.65,.27,.36,1],mass=0,
                                  contype=0,conaffinity=0)
        for geom in spec.geoms:
            if geom.name == "blade":
                geom.size = [.34,.004,.16]
                geom.rgba = [.81,.85,.88,1]
            if geom.name == "board":
                geom.rgba = [.64,.43,.25,1]
        self.world._neutral_keyframe.qpos=[]
        self.world._rebuild_neutral_keyframe()
        self.model,self.data = self.world.compile()
        self.key=self.model.key("neutral").id
        self.edge=self.model.site("edge").id
        self.pad_site=self.model.site("holding_tip").id
        self.pad_geom=self.model.geom("holding_pad").id
        self.blade_geom=self.model.geom("blade").id
        self.piece_geoms={self.model.geom(n+"_geom").id for n in self.piece_names}
        self.geom_piece={self.model.geom(n+"_geom").id:i for i,n in enumerate(self.piece_names)}
        self.piece_bodies=[self.model.body(n).id for n in self.piece_names]
        self.piece_qadr=[self.model.jnt_qposadr[self.model.joint(n+"_free").id] for n in self.piece_names]
        self.seams=[self.model.equality(f"slice_seam{i}").id for i in range(len(self.cut_planes))]
        self.pose_cache={}
        self.thin_ready=True
        self.reset()

    def solve_slice_pose(self,target):
        probe=mj.MjData(self.model)
        mj.mj_resetDataKeyframe(self.model,probe,self.key)
        lower=self.model.jnt_range[self.joints,0]+.001
        upper=self.model.jnt_range[self.joints,1]-.001
        def residual(q):
            probe.qpos[self.qadr]=q
            mj.mj_forward(self.model,probe)
            rotation=probe.site_xmat[self.edge].reshape(3,3)
            return np.r_[probe.site_xpos[self.edge]-target,
                         .04*(rotation-np.eye(3)).ravel(),.001*(q-self.neutral)]
        fit=least_squares(residual,np.clip(self.neutral,lower,upper),bounds=(lower,upper),max_nfev=250)
        if np.linalg.norm(residual(fit.x)[:3]) > .06:
            raise RuntimeError("Slicing target is outside the reachable knife workspace")
        return fit.x

    def reset(self,offset=0,profile=None,nudge=0):
        if not self.thin_ready:
            return super().reset(offset)
        self.profile=profile or dict(aim_bias=0,descent=0.7,press=0.02)
        mj.mj_resetDataKeyframe(self.model,self.data,self.key)
        self.data.eq_active[self.seams]=True
        for i in range(self.model.nu):
            joint=self.model.actuator_trnid[i,0]
            self.data.ctrl[i]=self.data.qpos[self.model.jnt_qposadr[joint]]
        self.onion_xy=self.path_xy+[0,offset]
        for adr in self.piece_qadr:
            self.data.qpos[adr+1]+=offset
        mj.mj_forward(self.model,self.data)
        self.initial_centers=self.data.xpos[self.piece_bodies].copy()
        self.initial_rotations=self.data.xmat[self.piece_bodies].reshape(-1,3,3).copy()
        self.nudge=nudge
        self.hold_enabled=True
        self.fracture_enabled=True
        self.seam_work=np.zeros(len(self.seams))
        self.events=[]
        self.cut=False
        self.current_pad_force=0.0
        self.first_pad_contact=None
        self.pad_contact_s=0.0
        self.peak_force=self.peak_hazard=self.max_slip=0.0
        self.aim_samples=[]
        self.previous=self.data.site_xpos[self.edge].copy()
        self.stroke_index=0
        self.stroke_time=0
        self.elapsed=0
        self.cycle=0.35+self.profile["descent"]+0.4
        self.duration=0.8+len(self.seams)*self.cycle+0.3
        key=(float(offset),*(float(self.profile[k]) for k in ["aim_bias","descent","press"]))
        if key not in self.pose_cache:
            raised,lowered=[],[]
            for y in self.cut_planes+offset+self.profile["aim_bias"]:
                raised.append(self.solve_slice_pose([self.path_xy[0],y,self.top_z+0.20]))
                lowered.append(self.solve_slice_pose([self.path_xy[0],y,self.board_z+0.008]))
            target=np.array([self.path_xy[0],offset+0.19,self.board_z+.36*np.sqrt(1-(.19/.28)**2)+0.07-self.profile["press"]])
            self.pose_cache[key]=(raised,lowered,self.solve_left(target))
        self.stroke_raised,self.stroke_lowered,self.left_hold=self.pose_cache[key]

    def step(self,t,chop=True):
        self.elapsed=float(t)
        index=int(np.clip((t-.8)//self.cycle,0,len(self.seams)-1))
        local=max(0,t-.8-index*self.cycle)
        self.stroke_index=index
        self.stroke_time=local
        up,down=self.stroke_raised[index],self.stroke_lowered[index]
        if t < .8:
            right=self.neutral+smooth(t/.7)*(self.stroke_raised[0]-self.neutral)
        elif local < .35:
            start=self.stroke_raised[max(0,index-1)]
            right=start+smooth(local/.35)*(up-start)
        elif local < .35+self.profile["descent"] and chop:
            right=up+smooth((local-.35)/self.profile["descent"])*(down-up)
        else:
            right=(down+smooth((local-.35-self.profile["descent"])/.4)*(up-down)) if chop else up
        self.data.ctrl[self.indices]=right
        self.data.ctrl[self.left_indices]=(self.left_neutral+smooth(t/.6)*(self.left_hold-self.left_neutral)
                                          if self.hold_enabled else self.left_neutral)
        for body in self.piece_bodies:
            self.data.xfrc_applied[body,:]=0
        if .7 <= t < .75:
            self.data.xfrc_applied[self.piece_bodies[-1],0]=self.nudge
        for _ in range(self.substeps):
            mj.mj_step(self.model,self.data)
            blade=pad=hazard=0.0
            contacted=set()
            for c in range(self.data.ncon):
                contact=self.data.contact[c]
                pair={int(contact.geom1),int(contact.geom2)}
                if self.blade_geom not in pair and self.pad_geom not in pair:
                    continue
                force=np.zeros(6)
                mj.mj_contactForce(self.model,self.data,c,force)
                normal=max(0,float(force[0]))
                for geom in pair & self.piece_geoms:
                    if self.blade_geom in pair:
                        blade+=normal
                        if normal > .001:
                            contacted.add(self.geom_piece[geom])
                    if self.pad_geom in pair and self.geom_piece[geom] >= index:
                        pad+=normal
                if pair=={self.blade_geom,self.pad_geom}:
                    hazard+=normal
            edge=self.data.site_xpos[self.edge].copy()
            dz=max(0,float(self.previous[2]-edge[2]))
            remainder=self.piece_bodies[index+1]
            rotation=self.data.xmat[remainder].reshape(3,3)
            seam_center=self.data.xpos[remainder]+rotation@np.array([0,self.cut_planes[index],0])
            normal=rotation[:,1]
            distance=abs(float(np.dot(edge-seam_center,normal)))
            tilt=float(np.arccos(np.clip(abs(normal@self.data.site_xmat[self.edge].reshape(3,3)[:,1]),0,1)))
            if blade > .001 and dz > 0:
                self.aim_samples.append(distance)
            self.peak_force=max(self.peak_force,blade)
            self.peak_hazard=max(self.peak_hazard,hazard)
            self.current_pad_force=pad
            if pad > .001:
                if self.first_pad_contact is None:
                    self.first_pad_contact=float(self.data.time)
                self.pad_contact_s+=self.model.opt.timestep
            drift=float(np.linalg.norm(self.data.xpos[self.piece_bodies[-1],:2]-self.initial_centers[-1,:2]))
            self.max_slip=max(self.max_slip,drift)
            target_contact=bool(contacted & {index,index+1})
            if self.data.eq_active[self.seams[index]] and distance < .025 and tilt < .22 and target_contact:
                self.seam_work[index]+=blade*dz
                if (self.fracture_enabled and self.seam_work[index] >= .01 and dz > 0
                        and edge[2] < seam_center[2]+self.radius[2]*np.sqrt(1-(self.cut_planes[index]/.28)**2)-.01
                        and blade > .001):
                    self.data.eq_active[self.seams[index]]=False
                    self.events.append(dict(seam=index,time_s=float(self.data.time),alignment_error_mm=distance,
                        tilt_degrees=float(np.degrees(tilt)),pad_force=pad,slip_mm=drift,blade_force=blade,
                        thickness_mm=float(self.boundaries[index+1]-self.boundaries[index]),
                        contact_work=float(self.seam_work[index]),full_depth=False))
                    self.cut=True
            for event in self.events:
                if event["seam"]==index and edge[2] <= self.board_z+.045 and dz > 0:
                    event["full_depth"]=True
            self.previous=edge
        mj.mj_forward(self.model,self.data)
        if not np.isfinite(self.data.qpos).all() or np.max(np.abs(self.data.qvel))>1e6:
            raise RuntimeError("Unstable slicing physics; reject trial")

    def metrics(self):
        # A slab is detached only if all of its original links have broken.
        detached=[i for i in range(len(self.cut_planes))
                  if not self.data.eq_active[self.seams[i]]
                  and (i==0 or not self.data.eq_active[self.seams[i-1]])]
        clean=[e for e in self.events if e["seam"] in detached and e["full_depth"] and e["alignment_error_mm"] <= .018
               and e["tilt_degrees"] <= 8 and e["pad_force"]>.001 and e["slip_mm"]<.08]
        return dict(slices_detached=len(detached),clean_slices=len(clean),target_slices=len(self.seams),
                    seams_broken=len(self.events),events=self.events,peak_blade_force=self.peak_force,
                    blade_foot_force=self.peak_hazard,max_remainder_slip_mm=self.max_slip,
                    pad_contact_s=self.pad_contact_s,target_thickness_mm=self.slice_width,
                    fracture_model="rigid predefined seams; clean is a contact/alignment proxy")
