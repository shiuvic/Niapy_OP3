#!/usr/bin/env python
import time
from threading import Thread
import numpy as np
import pybullet as p
import datetime
from core.op3 import OP3
from walking.wfunc import WFunc
import sys

PYTHONUNBUFFERED = 1
if sys.platform == "win32":
    from ctypes import windll

    timeBeginPeriod = windll.winmm.timeBeginPeriod
    timeBeginPeriod(1)

class Walker(OP3):
    """
    Class for making Darwin walk
    """

    def __init__(self, x_vel=1, y_vel=0, ang_vel=0, interval=0.0054, *args, **kwargs):
        OP3.__init__(self, *args, **kwargs)

        self.running = False
        self.velocity = [0, 0, 0]
        self.walking = False
        self.fall = False
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.ang_vel = ang_vel
        self.current_velocity = [0, 0, 0]
        self.sld_x_vel = p.addUserDebugParameter("x_vel", -10, 10, x_vel)
        self.sld_y_vel = p.addUserDebugParameter("y_vel", -10, 10, y_vel)
        self.sld_ang_vel = p.addUserDebugParameter("ang_vel", -10, 10, ang_vel)

        self.wfunc = WFunc()
        # ~ self.ready_pos=get_walk_angles(10)
        self.ready_pos = self.wfunc.get(True, 0, [0, 0, 0])
        self.ready_pos.update({"r_sho_pitch": 0, "l_sho_pitch": 0,
                               "r_sho_roll": -1.0, "l_sho_roll": 1.0,
                               "r_el": 0.5, "l_el": -0.5})
        self._th_walk = None

        self.sld_interval = p.addUserDebugParameter("step_interval", 0.001, 0.01, interval)
        self.save_button = p.addUserDebugParameter("save parameters", 1, -1, 1)
        self.check_gui_th()
        self.stop_count = 0
        self.time__()
    def update_new_vel(self, x_vel, y_vel, ang_vel, parm,offset):
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.ang_vel = ang_vel
        self.wfunc._update_new_param(parm,offset)


    def robot_postion(self):
        position, _ = p.getBasePositionAndOrientation(self.robot)
        return np.array(position)


    def cmd_vel(self, vx, vy, vt):
        print("cmdvel", (vx, vy, vt))
        self.start()
        self.set_velocity(vx, vy, vt)

    def init_walk(self):
        """
        If not there yet, go to initial walk position
        """
        if self.get_dist_to_ready() > 0.02:
            time.sleep(2)
            self.set_angles_slow(self.ready_pos)

    def start(self):
        if not self.running:
            print("Start Walking", flush=True)
            self.fall = False
            self.running = True
            self.init_walk()
            self._th_walk = Thread(target=self._do_walk)
            self._th_walk.start()
            self.timemer = datetime.datetime.now()
            self.walking = True

    def stop(self):
        if self.running:
            self.fall = True
            self.walking = False
            print("Waiting for stopped")
            time.sleep(1)
            while self._th_walk is not None:
                time.sleep(0.1)
            print("Stopped")
            self.running = False
            time.sleep(1)

    def set_velocity(self, x, y, t):
        self.velocity = [x, y, t]

    def check_gui_th(self):
        def check_gui():
            while True:
                curr_x_vel = p.readUserDebugParameter(self.sld_x_vel)
                curr_y_vel = p.readUserDebugParameter(self.sld_y_vel)
                curr_ang_vel = p.readUserDebugParameter(self.sld_ang_vel)
                if self.x_vel != curr_x_vel or \
                        self.y_vel != curr_y_vel or \
                        self.ang_vel != curr_ang_vel:
                    self.x_vel = curr_x_vel
                    self.y_vel = curr_y_vel
                    self.ang_vel = curr_ang_vel
                    self.velocity = [self.x_vel, self.y_vel, self.ang_vel]
                self.camera_follow(distance=0.5)
                time.sleep(0.01)
        Thread(target=check_gui).start()

    def _do_walk(self):
        """
        Main walking loop, smoothly update velocity vectors and apply corresponding angles
        """
        # Global walk loop
        n = 60
        phrase = True
        i = 0
        self.current_velocity = [0, 0, 0]
        self.stop_count = 0
        while self.walking or i < n or self.is_walking():
            if not self.walking:
                self.velocity = [0, 0, 0]
            elif not self.is_walking() and i == 0:  # Do not move if nothing to do and already at 0
                self.update_velocity(self.velocity, n)
                time.sleep(p.readUserDebugParameter(self.sld_interval) / self.sim_speed)
                continue
            x = float(i) / n
            angles = self.wfunc.get(phrase, x, self.current_velocity)
            self.update_velocity(self.velocity, n)
            self.set_angles(angles)

            i += 1
            if i > n:
                i = 0
                phrase = not phrase
            time.sleep(p.readUserDebugParameter(self.sld_interval) / self.sim_speed)
            self.stop_count += 1
        self._th_walk = None

    def is_walking(self):
        e = 0.02
        for v in self.current_velocity:
            if abs(v) > e: return True
        return False

    def rescale(self, angles, coef):
        z = {}
        for j, v in angles.items():
            offset = self.ready_pos[j]
            v -= offset
            v *= coef
            v += offset
            z[j] = v
        return z

    def update_velocity(self, target, n):
        a = 3 / float(n)
        b = 1 - a
        self.current_velocity = [a * t + b * v for (t, v) in zip(target, self.current_velocity)]

    def get_dist_to_ready(self):
        angles = self.get_angles()
        return get_distance(self.ready_pos, angles)

    def reset_and_start(self):
        self.stop()
        p.resetBasePositionAndOrientation(self.robot, self.op3StartPos, self.op3StartOrientation)
        self.start()
        self.set_velocity(self.x_vel, self.y_vel, self.ang_vel)

    def time__(self):
        def time_th():
            while True:
                if(self.stop_count >= 1000):
                    self.reset_and_start()
                time.sleep(0.5)
        Thread(target=time_th).start()


def interpolate(anglesa, anglesb, coefa):
    z = {}
    joints = anglesa.keys()
    for j in joints:
        z[j] = anglesa[j] * coefa + anglesb[j] * (1 - coefa)
    return z


def get_distance(anglesa, anglesb):
    d = 0
    joints = anglesa.keys()
    if len(joints) == 0: return 0
    for j in joints:
        d += abs(anglesb[j] - anglesa[j])
    d /= len(joints)
    return d


if __name__ == "__main__":

    walker = Walker(fallen_reset=True)
    time.sleep(1)
    walker.reset_and_start()

    while True:
        time.sleep(1)
