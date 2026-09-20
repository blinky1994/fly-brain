"""Activity-only adapter for our MaleCNS worker; never runs a second brain.

Projection and NumPy glow rendering adapted from NullLabTests/FLYBOARD's
flyboard/render.py (MIT, copyright 2026 NullLabTests). See third_party/flyboard/LICENSE.
Changes: measured local atlas, spike-count-only brightness, missing-position
accounting, visibility-correct picking, GLFW/Pillow UI and same-worker telemetry.
"""
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter, uniform_filter
from neuron_activity import ActivityState


class NeuronRenderer:
    def __init__(self,atlas):
        self.activity=ActivityState(atlas)
        a=self.activity
        self.xyz=np.zeros_like(a.xyz)
        points=a.xyz[a.positioned]
        center=(points.min(axis=0)+points.max(axis=0))/2
        scale=np.ptp(points,axis=0).max()/2
        # Rigid rotation: EM z runs down the displayed nerve cord. One uniform scale.
        self.xyz[a.positioned]=(points-center)/scale @ np.array([[1,0,0],[0,0,1],[0,-1,0]])
        self.yaw=self.pitch=0.
        self.zoom=1.
        self.mode=0
        self.selected=None
        self.context='Waiting for the neural worker'
        self.connected=True
        self.last_frame=None
        self.projection=None
        self.colors=np.tile([.18,.44,.58],(len(a.body_ids),1))
        optic=np.array(['ol_' in c or 'visual' in c for c in a.classes])
        self.colors[optic]=[.36,.25,.51]
        self.colors[a.motor]=[.12,.64,.43]
        self.colors[a.sensor]=[.23,.48,.78]
        try:
            self.font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',16)
            self.small=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',13)
            self.title=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',23)
        except OSError:
            self.font=self.small=self.title=ImageFont.load_default()

    def update(self,frame):
        self.activity.update(frame)
        self.last_frame=time.monotonic()

    def clear(self,context):
        self.activity.clear()
        self.context=context
        self.last_frame=None

    def visible_mask(self):
        a=self.activity
        keep=a.positioned.copy()
        if self.mode==1:
            keep &= np.array(['vnc' in c or c in ['ascending_neuron','sensory_ascending'] for c in a.classes]) | a.motor | a.sensor
        elif self.mode==2:
            keep &= a.motor | a.sensor
        return keep

    def project(self,width,height):
        cy,sy=np.cos(self.yaw),np.sin(self.yaw)
        cp,sp=np.cos(self.pitch),np.sin(self.pitch)
        rotation=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
        p=self.xyz @ rotation.T
        scale=max(100,height-230)*.43*self.zoom
        u=width*.5+p[:,0]*scale
        v=height*.5+25-p[:,1]*scale
        keep=self.visible_mask() & (u>=0) & (u<width) & (v>=130) & (v<height-95)
        self.projection=(u,v,keep)
        return u,v,keep

    def pick(self,x,y):
        if self.projection is None: return
        u,v,keep=self.projection
        distance=(u-x)**2+(v-y)**2
        # Prefer a nearby firing cell over overlapping dim anatomy.
        candidates=np.flatnonzero(keep & (self.activity.counts>0) & (distance<12**2))
        if not len(candidates): candidates=np.flatnonzero(keep & (distance<7**2))
        self.selected=int(candidates[np.argmin(distance[candidates])]) if len(candidates) else None

    def render(self,width=1040,height=820):
        a=self.activity
        u,v,keep=self.project(width,height)
        index=np.flatnonzero(keep)
        flat=v[index].astype(int)*width+u[index].astype(int)
        background=np.zeros((height*width,3),dtype=np.float32)
        for c in range(3): np.maximum.at(background[:,c],flat,self.colors[index,c]*.55)
        img=background.reshape(height,width,3)
        img=maximum_filter(img,size=(2,2,1))
        # Frames contain counts over a declared window, not instantaneous spikes.
        # A fixed 5-spikes/window scale, never a per-frame maximum normalization.
        fresh=self.connected and self.last_frame is not None and time.monotonic()-self.last_frame<3
        active=np.flatnonzero(keep & (a.counts>0)) if fresh else np.array([],dtype=int)
        glow=np.zeros((height,width,3),dtype=np.float32)
        if len(active):
            pixels=v[active].astype(int)*width+u[active].astype(int)
            brightness=.45+.55*np.minimum(a.counts[active]/5,1)
            for c,color in enumerate([1.,.59,.12]):
                np.maximum.at(glow.reshape(-1,3)[:,c],pixels,brightness*color)
            glow=maximum_filter(glow,size=(5,5,1))
            img += glow+uniform_filter(glow,size=(9,9,1))*.7
        rgb=np.clip(img*255+np.array([9,15,25]),0,255).astype(np.uint8)
        image=Image.fromarray(rgb)
        draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,width,127),fill=(13,23,36))
        draw.text((22,12),'MaleCNS neuron activity',font=self.title,fill=(233,242,250))
        status='LIVE - simulated spikes' if fresh else ('Waiting for next neural frame' if self.connected else 'Disconnected - activity cleared')
        draw.text((22,45),status,font=self.font,fill=(255,190,95) if fresh else (165,185,205))
        draw.text((22,71),self.context,font=self.small,fill=(177,200,218))
        active_all=int(np.count_nonzero(a.counts)) if fresh else 0
        missing=int(np.count_nonzero(a.counts[~a.positioned])) if fresh else 0
        line=(f'{active_all:,} firing cells / {len(a.body_ids):,} simulated | '
              f'{missing} firing without positions | neural time {a.time_ms:.0f} ms | {a.window_ms:g} ms window')
        draw.text((22,96),line,font=self.small,fill=(177,200,218))
        if self.selected is not None:
            i=self.selected
            if keep[i]: draw.ellipse((u[i]-8,v[i]-8,u[i]+8,v[i]+8),outline=(245,248,255),width=2)
            label=f'ID {a.body_ids[i]} | {a.types[i]} | {a.classes[i]} | side {a.sides[i]}'
            draw.rectangle((12,height-153,width-12,height-101),fill=(20,34,50))
            draw.text((22,height-148),label,font=self.small,fill=(235,242,248))
            draw.text((22,height-125),f'{int(a.counts[i]) if fresh else 0} spikes in last displayed window | '
                      +('motor output' if a.motor[i] else 'sensory input' if a.sensor[i] else 'network neuron'),
                      font=self.small,fill=(255,194,105))
        draw.rectangle((0,height-94,width,height),fill=(13,23,36))
        draw.text((22,height-87),f'{int(a.positioned.sum()):,} measured soma positions; '
                  f'{int((~a.positioned).sum()):,} unpositioned | View: {self.mode+1}',font=self.small,fill=(165,189,206))
        draw.text((22,height-65),'Drag: rotate   Wheel: zoom   Click: inspect   1: whole CNS   2: nerve cord   3: motor/sensory   R: reset',
                  font=self.small,fill=(210,224,235))
        draw.text((22,height-43),'Orange = fired; dim dots = anatomy only. Cell bodies, not neuron branches or synaptic edges.',
                  font=self.small,fill=(210,224,235))
        draw.text((22,height-22),'Renderer adapted from FLYBOARD / NullLabTests (MIT) | MaleCNS: FlyEM / Janelia + collaborators',
                  font=self.small,fill=(135,159,180))
        return np.asarray(image)


class NeuronViewer(NeuronRenderer):
    def __init__(self,atlas):
        super().__init__(atlas)
        import glfw
        self.glfw=glfw
        if not glfw.init(): raise RuntimeError('Cannot initialize neuron viewer')
        self.window=glfw.create_window(1040,820,'MaleCNS - live neuron activity',None,None)
        if not self.window: raise RuntimeError('Cannot open neuron viewer')
        self.last_render=0.
        self.drag_start=self.cursor=None
        glfw.set_key_callback(self.window,self._key)
        glfw.set_scroll_callback(self.window,lambda w,x,y:setattr(self,'zoom',max(.3,min(8,self.zoom*1.1**y))))
        glfw.set_mouse_button_callback(self.window,self._mouse)

    def _key(self,w,key,scan,action,mods):
        g=self.glfw
        if action!=g.PRESS: return
        if key in [g.KEY_1,g.KEY_2,g.KEY_3]: self.mode=key-g.KEY_1
        if key==g.KEY_R: self.yaw=self.pitch=0.; self.zoom=1.
        if key in [g.KEY_Q,g.KEY_ESCAPE]: g.set_window_should_close(self.window,True)

    def _mouse(self,w,button,action,mods):
        g=self.glfw
        if button!=g.MOUSE_BUTTON_LEFT: return
        p=np.array(g.get_cursor_pos(w))
        if action==g.PRESS: self.drag_start=p; self.cursor=p
        elif self.drag_start is not None:
            if np.linalg.norm(p-self.drag_start)<5:
                fw,fh=g.get_framebuffer_size(w); ww,wh=g.get_window_size(w)
                self.pick(p[0]*fw/max(ww,1),p[1]*fh/max(wh,1))
            self.drag_start=self.cursor=None

    def running(self):
        return self.window is not None and not self.glfw.window_should_close(self.window)

    def sync(self):
        if self.window is None: return
        g=self.glfw
        g.poll_events()
        if not self.running():
            self.close()
            return
        if self.cursor is not None:
            p=np.array(g.get_cursor_pos(self.window)); delta=p-self.cursor
            self.yaw+=delta[0]*.006; self.pitch+=delta[1]*.006; self.cursor=p
        if time.monotonic()-self.last_render<.10: return
        w,h=g.get_framebuffer_size(self.window)
        if w<1 or h<1: return
        from OpenGL import GL
        g.make_context_current(self.window)
        rgb=self.render(w,h)
        GL.glViewport(0,0,w,h)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glMatrixMode(GL.GL_PROJECTION); GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW); GL.glLoadIdentity()
        GL.glRasterPos2f(-1,-1)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
        GL.glDrawPixels(w,h,GL.GL_RGB,GL.GL_UNSIGNED_BYTE,np.ascontiguousarray(rgb[::-1]))
        g.swap_buffers(self.window)
        self.last_render=time.monotonic()

    def close(self):
        self.connected=False
        self.activity.clear()
        if self.window is not None:
            self.glfw.destroy_window(self.window)
            self.window=None
