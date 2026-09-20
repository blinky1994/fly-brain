"""Train existing MaleCNS efficacies through continuous motor-neuron body feedback."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from intact_onion import IntactOnion

ROOT=Path(__file__).resolve().parent


class MotorClient:
    def __init__(self,out):
        self.log=(out/'neural_worker.log').open('w')
        self.process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'malecns_motor_brain.py')],
            cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)

    def request(self,payload):
        self.process.stdin.write(json.dumps(payload)+'\n')
        self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:
            raise RuntimeError('Neural worker stopped; see neural_worker.log')
        result=json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lesson',choices=['reach','slice'],default='reach',
                   help='Start with a fixed knife-tip reach; slice retains the older combined objective')
    p.add_argument('--episodes',type=int,default=12)
    p.add_argument('--seconds',type=float,default=3)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--neurons',action='store_true',help='Show live per-neuron activity beside the body viewer')
    p.add_argument('--record-neurons',action='store_true',help='Record per-neuron frames even without a viewer')
    p.add_argument('--continuous',action='store_true')
    p.add_argument('--frozen',action='store_true')
    p.add_argument('--silence-motor-output',action='store_true',help='Ablation: motor spikes cannot command movement')
    p.add_argument('--speed',type=float,default=1)
    args=p.parse_args()
    if args.episodes<1 or not np.isfinite([args.seconds,args.speed]).all() or args.seconds<.1 or args.speed<0:
        p.error('Use positive episodes/duration and a nonnegative speed')
    if args.continuous and args.headless:
        p.error('Continuous mode requires a viewer; closing it stops learning')
    if args.neurons and args.headless:
        p.error('Use --record-neurons for headless recording, or omit --headless to open the neuron viewer')
    if args.silence_motor_output and not args.frozen:
        p.error('Use --frozen with the motor-output ablation')
    out=(args.output_dir or ROOT/'data/experiments/connectome_motor'/str(time.time_ns())).resolve()
    if not out.is_relative_to(ROOT) or (out.exists() and any(out.iterdir())):
        p.error('Use a fresh output directory inside this project')
    out.mkdir(parents=True,exist_ok=True)
    config={**vars(args),'checkpoint':str(args.checkpoint) if args.checkpoint else None,'output_dir':str(out),
            'scope':'Full MaleCNS LIF; annotated motor outputs; provisional body mapping; reward-searched internal efficacies',
            'control_seconds':.1,'neural_ms_per_control':20,'cutting':'custom dynamic planar split; not tissue FEM'}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print('Session:',out,flush=True)
    if args.lesson == 'reach':
        from reaching_lesson import ReachingLesson
        scene=ReachingLesson()
    else:
        scene=IntactOnion()
    viewer=None
    neurons=None
    brain=None
    pool=ThreadPoolExecutor(max_workers=1)
    results=[]
    try:
        if not args.headless:
            from motor_viewer import MotorViewer
            viewer=MotorViewer(scene,args.speed)
            viewer.sync(scene)
        brain=MotorClient(out)

        def request(operation,**payload):
            future=pool.submit(brain.request,dict(operation=operation,**payload))
            deadline=time.monotonic()+180
            while not future.done():
                if viewer:
                    if not viewer.running(): raise InterruptedError
                    viewer.sync(scene)
                if neurons: neurons.sync()
                if time.monotonic()>deadline: raise TimeoutError('Neural request exceeded 180 seconds')
                time.sleep(.01)
            return future.result()

        metadata=request('init',motor_names=scene.motor_names,observation_size=len(scene.observation()),
                         checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None)
        (out/'neural_model.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata),flush=True)
        if args.neurons or args.record_neurons:
            atlas=request('viewer_atlas',path=str(out/'neuron_atlas.npz'))
            (out/'neuron_atlas.json').write_text(json.dumps(atlas,indent=2))
            if args.neurons:
                from neuron_viewer import NeuronViewer
                neurons=NeuronViewer(atlas['path'])
                neurons.sync()

        def trial(phase,episode,learn=False,offset=0,reference=False):
            scene.reset(offset)
            if neurons: neurons.clear(f'{phase} {episode} | learning: {learn} | same worker as body')
            request('begin',learn=learn,silence=args.silence_motor_output)
            trajectory=[]
            for command in range(max(1,round(args.seconds/.1))):
                if viewer:
                    viewer.text=(f'MaleCNS direct motor learning | {phase} {episode}\n'
                        f'Learning: {learn} | physical pieces: {len(scene.pieces)} | cuts: {len(scene.events)}\n'
                        'Computing fresh neural activity...\nDrag: orbit | scroll: zoom | 1/2/3/0: speed | close: stop')
                neural=request('act',observation=scene.observation().tolist(),
                               capture_activity=args.neurons or args.record_neurons)
                if neurons:
                    neurons.update(neural)
                    neurons.context=f'{phase} {episode} | learning: {learn} | body time {scene.data.time:.1f} s'
                trajectory.append(dict(time=float(scene.data.time),action=neural['action'],
                                       motor_spikes=neural['motor_spikes'],spikes=neural['spikes'],
                                       motor_activity=neural['motor_activity']))
                if 'neuron_spikes' in neural:
                    trajectory[-1].update({k:neural[k] for k in ['neuron_spikes','neural_time_ms','window_ms']})
                for _ in range(5):
                    tick=time.monotonic()
                    scene.step(neural['action'])
                    if viewer:
                        if not viewer.running(): raise InterruptedError
                        viewer.text=(f'MaleCNS | {phase} {episode} | learning: {learn}\n'
                            f'Neuron spikes: {neural["spikes"]} | motor spikes: {neural["motor_spikes"]}\n'+
                            (f'Target distance: {scene.distance():.3f} mm | hold: {scene.dwell:.2f} / .20 s\n'
                             if args.lesson == 'reach' else
                             f'Physical pieces: {len(scene.pieces)} | cuts: {len(scene.events)}\n')+
                            'Direct joint commands; no movement library\nDrag: orbit | scroll: zoom | 1/2/3/0: speed')
                        viewer.sync(scene)
                        if neurons: neurons.sync()
                        if viewer.speed:
                            time.sleep(max(0,scene.dt/viewer.speed-(time.monotonic()-tick)))
            result=request('finish',reward=scene.score(),reference=reference)
            actions=np.asarray([c['action'] for c in trajectory])
            result.update(phase=phase,episode=episode,offset=offset,physics=scene.metrics(),
                          left_command_peak=float(np.max(np.abs(actions[:,:7]))),
                          right_command_peak=float(np.max(np.abs(actions[:,7:]))),
                          joint_displacement=float(np.linalg.norm(scene.data.qpos[scene.motor_qadr]-scene.neutral_motor)))
            results.append(result)
            with (out/'trials.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
            with (out/'activity.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(phase=phase,episode=episode,commands=trajectory))+'\n')
            request('save',path=str(out/'latest.npz'))
            print(json.dumps(result),flush=True)
            return result

        baseline=trial('baseline',0,reference=True)
        block=0
        while True:
            for i in range(args.episodes): trial('training',block*args.episodes+i+1,not args.frozen)
            evaluation=[trial('evaluation',block*3+i+1,offset=offset) for i,offset in enumerate([0,-.006,.006])]
            report=dict(baseline=baseline,evaluation=evaluation,blocks=block+1,
                        lesson=args.lesson,
                        reaching_demonstrated=(args.lesson=='reach' and
                            all(r['physics']['reach_success'] for r in evaluation)),
                        checkpoint=str(out/'latest.npz'),scope=config['scope'],
                        clean_slicing_demonstrated=all(r['physics']['clean_slices']>=3 for r in evaluation))
            (out/'report.json').write_text(json.dumps(report,indent=2))
            print('Report:',out/'report.json',flush=True)
            block+=1
            results.clear()
            if not args.continuous: break
    except (KeyboardInterrupt,InterruptedError):
        print('Stopped. The last completed neural update is saved.',flush=True)
    finally:
        if brain: brain.close()
        pool.shutdown(wait=True,cancel_futures=True)
        if neurons: neurons.close()
        if viewer: viewer.close()


if __name__=='__main__':
    main()
