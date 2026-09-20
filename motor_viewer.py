"""MuJoCo window that survives model recompilation when a new cut creates bodies."""
import glfw
import mujoco as mj


class MotorViewer:
    def __init__(self, scene, speed=1):
        if not glfw.init():
            raise RuntimeError('Cannot initialize the interactive viewer')
        self.window = glfw.create_window(1200,850,'MaleCNS - intact onion / direct motor learning',None,None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('Cannot open the viewer window')
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.camera = mj.MjvCamera()
        self.camera.lookat[:] = [*scene.path_xy,scene.board_z+.18]
        self.camera.distance,self.camera.azimuth,self.camera.elevation = 2.8,135,-30
        self.option = mj.MjvOption()
        self.context = self.scene = None
        self.revision = -1
        self.speed = speed
        self.text = 'Loading full MaleCNS connectome...'
        self.last_cursor = None
        glfw.set_key_callback(self.window,self._key)
        glfw.set_scroll_callback(self.window,lambda w,x,y:setattr(self.camera,'distance',
            max(.4,min(10,self.camera.distance*(.9**y)))))

    def _key(self,window,key,scancode,action,mods):
        if action != glfw.PRESS:
            return
        if key in [glfw.KEY_ESCAPE,glfw.KEY_Q]:
            glfw.set_window_should_close(window,True)
        if key in [glfw.KEY_0,glfw.KEY_1,glfw.KEY_2,glfw.KEY_3]:
            self.speed={glfw.KEY_0:0,glfw.KEY_1:.5,glfw.KEY_2:1,glfw.KEY_3:4}[key]

    def running(self):
        return not glfw.window_should_close(self.window)

    def sync(self,environment):
        glfw.make_context_current(self.window)
        glfw.poll_events()
        if self.revision!=environment.revision:
            if self.context:
                self.context.free()
            self.context=mj.MjrContext(environment.model,mj.mjtFontScale.mjFONTSCALE_150)
            self.scene=mj.MjvScene(environment.model,maxgeom=10000)
            self.revision=environment.revision
        cursor=glfw.get_cursor_pos(self.window)
        if self.last_cursor and glfw.get_mouse_button(self.window,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS:
            self.camera.azimuth += .3*(cursor[0]-self.last_cursor[0])
            self.camera.elevation = max(-89,min(0,self.camera.elevation+.3*(cursor[1]-self.last_cursor[1])))
        self.last_cursor=cursor
        viewport=mj.MjrRect(0,0,*glfw.get_framebuffer_size(self.window))
        mj.mjv_updateScene(environment.model,environment.data,self.option,None,self.camera,
                           mj.mjtCatBit.mjCAT_ALL,self.scene)
        mj.mjr_render(viewport,self.scene,self.context)
        mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL,mj.mjtGridPos.mjGRID_TOPLEFT,
                       viewport,self.text,'',self.context)
        glfw.swap_buffers(self.window)

    def close(self):
        glfw.make_context_current(self.window)
        if self.context:
            self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()
