"""Scripted fixture for cutting mechanics ONLY; never used by neural training."""
import json
from pathlib import Path
import time
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
import imageio.v3 as imageio
from intact_onion import IntactOnion, BASE


def pose(scene,indices,site,target,level=False):
    joints=scene.model.actuator_trnid[indices,0]
    addresses=scene.model.jnt_qposadr[joints]
    probe=mj.MjData(scene.model)
    probe.qpos[:]=scene.data.qpos
    initial=probe.qpos[addresses].copy()
    limits=scene.model.jnt_range[joints]
    def residual(q):
        probe.qpos[addresses]=q
        mj.mj_forward(scene.model,probe)
        r=probe.site_xmat[site].reshape(3,3)
        return np.r_[probe.site_xpos[site]-target,
                     .3*(r[:,1]-[0,1,0]) if level else [],
                     [r[2,0]] if level else [], .0001*(q-initial)]
    fitted=least_squares(residual,np.clip(initial,limits[:,0]+.001,limits[:,1]-.001),
                         bounds=(limits[:,0]+.001,limits[:,1]-.001),max_nfev=300)
    if np.linalg.norm(residual(fitted.x)[:3])>.04:
        raise RuntimeError('Unreachable scripted fixture pose')
    return fitted.x


def main():
    out=Path('data/experiments/intact_cut_checks')/str(time.time_ns())
    out.mkdir(parents=True)
    s=IntactOnion()
    camera=mj.MjvCamera()
    camera.lookat[:]=[*s.path_xy,s.board_z+.18]
    camera.distance,camera.azimuth,camera.elevation=1.8,135,-30
    def render(name):
        with mj.Renderer(s.model,height=700,width=1000) as r:
            r.update_scene(s.data,camera=camera)
            imageio.imwrite(out/name,r.render())
    render('before.png')
    right=np.array([i for i in s.motor_indices if 'rf_' in s.model.actuator(i).name])
    left=np.array([i for i in s.motor_indices if 'lf_' in s.model.actuator(i).name])
    center=s.center.copy()
    up=center+[0,-.209,.28]
    right_path=np.array([pose(s,right,s.edge,center+[0,-.209,z],True)
                         for z in np.linspace(.28,BASE+.003,17)])
    holding=center+[0,.1,.18*np.sqrt(1-(.1/.28)**2)+.07-.016]
    hover=holding+[0,0,.15]
    start=s.data.site_xpos[s.pad_site].copy()
    lift=start.copy(); lift[2]=hover[2]
    left_path=np.array([pose(s,left,s.pad_site,t) for t in
                       np.vstack([np.linspace(start,lift,5),np.linspace(lift,hover,9)[1:],np.linspace(hover,holding,7)[1:]])])
    right_start=s.data.ctrl[right].copy()
    def follow(path,u):
        p=np.clip(u,0,1)*(len(path)-1)
        k=min(int(p),len(path)-2)
        return path[k]+(p-k)*(path[k+1]-path[k])
    trace=[]
    for t in np.arange(0,4,s.dt):
        s.data.ctrl[left]=follow(left_path,t/1.3)
        if t<1.5:
            s.data.ctrl[right]=right_start+min(1,t/.9)*(right_path[0]-right_start)
        elif t<3:
            s.data.ctrl[right]=follow(right_path,(t-1.5)/1.5)
        else:
            s.data.ctrl[right]=follow(right_path,1-(t-3))
        s.step(np.zeros(14))
        if round(t/s.dt)%5==0:
            body=s.model.body(s.remainder()).id
            trace.append(dict(time=float(t),incision=dict(s.inc) if s.inc else None,
                              local_edge=(s.data.xmat[body].reshape(3,3).T @
                              (s.data.site_xpos[s.edge]-s.data.xpos[body])).tolist()))
    render('after.png')
    report=dict(scope='Scripted mechanics fixture; NOT learned movements',**s.metrics(),
                pending_incision=s.inc)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'trace.json').write_text(json.dumps(trace,indent=2))
    print(json.dumps(report),flush=True)
    print('Artifacts:',out,flush=True)
    assert report['cuts']==1,'Scripted full-depth contact stroke did not cut'
    assert report['physical_pieces']==2,'Expected dynamically created pieces'


if __name__=='__main__':
    main()
