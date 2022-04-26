import math
import time
from threading import Thread

import pybullet as p

from walking.wj_func import WJFunc


class WFunc:
    """
    Multi-joint walk function for Darwin
    """

    def __init__(self, **kwargs):
        walk_offset = {'hip_pitch': -0.063,
                       'hip_roll': 0.0,
                       'hip_yaw': 0.0,
                       'ank_pitch': 0.0,
                       'ank_roll': 0.0,
                       'knee': 0.0}

        self.parameters = {"swing_scale": 0.0,
                            "step_scale": 0.1,
                            "step_offset": 0.25,
                            "ankle_offset": 0.0,
                            "vx_scale": 0.2,
                            "vy_scale": 0.2,
                            "vt_scale": 0.1}
        self.ang_offsets = {}
        self.param_sliders = {}
        self.ang_sliders = {}
        for k, v in self.parameters.items():
            self.param_sliders[k] = p.addUserDebugParameter(k, -3, 3, v)
        for k, v in walk_offset.items():
            self.ang_sliders[k] = p.addUserDebugParameter(k, -3, 3, v)
        # self.update_param_th()
        self.update_param()
        while not self.ang_offsets:
            print("wait parameters to propagated")
        for k, v in kwargs.items():
            self.parameters[k] = v
        self.generate_init()

    def update_param_th(self):
        def _update_param():
            while True:
                self.update_param()

                time.sleep(0.001)
        Thread(target=_update_param).start()

    def update_param(self):
        for k, v in self.param_sliders.items():
            self.parameters[k] = -p.readUserDebugParameter(v)
        for k, v in self.ang_sliders.items():
            offset = p.readUserDebugParameter(v)
            self.ang_offsets["l_" + k] = offset
            self.ang_offsets["r_" + k] = -offset

    def _update_new_param(self, param_pso, ang_offsets_pso):
        for k,v in param_pso.items():
            self.parameters[k] = -v
        for k,v in ang_offsets_pso.items():
            offset = v
            self.ang_offsets["l_" + k] = offset
            self.ang_offsets["r_" + k] = -offset



    def generate_init(self):
        """
        Build CPG functions for walk-on-spot (no translation or rotation, only legs up/down)
        """
        # f1=THIGH1=ANKLE1=L=R in phase
        self.pfn = {}  # phase joint functions
        self.afn = {}  # anti phase joint functions

        # ~ print f
        f1 = WJFunc()
        f1.in_scale = math.pi
        f1.scale = -self.parameters["swing_scale"]
        self.pfn["l_ank_roll"] = f1
        self.pfn["l_hip_roll"] = f1

        # f2=mirror f1 in antiphase
        f2 = f1.mirror()
        # ~ f2=WJFunc()
        self.afn["l_ank_roll"] = f2
        self.afn["l_hip_roll"] = f2

        f3 = WJFunc()
        f3.in_scale = math.pi
        f3.scale = self.parameters["step_scale"]
        f3.offset = self.parameters["step_offset"]
        self.pfn["l_hip_pitch"] = f3
        f33 = f3.mirror()
        f33.offset += self.parameters["ankle_offset"]
        self.pfn["l_ank_pitch"] = f33

        f4 = f3.mirror()
        f4.offset *= 2
        f4.scale *= 2
        self.pfn["l_knee"] = f4

        f5 = f3.clone()
        f5.in_scale *= 2
        f5.scale = 0
        self.afn["l_hip_pitch"] = f5

        f6 = f3.mirror()
        f6.in_scale *= 2
        f6.scale = f5.scale
        f6.offset += self.parameters["ankle_offset"]
        self.afn["l_ank_pitch"] = f6

        f7 = f4.clone()
        f7.scale = 0
        self.afn["l_knee"] = f7

        self.forward = [f5, f6]

        self.generate_right_init()
        self.joints = self.pfn.keys()

        self.show()

    def generate_right_init(self):
        """
        Mirror CPG functions from left to right and antiphase right
        """
        l = [v[2:] for v in self.pfn.keys()]
        for j in l:
            self.pfn["r_" + j] = self.afn["l_" + j].mirror()
            self.afn["r_" + j] = self.pfn["l_" + j].mirror()

    def generate(self):
        # ~ print f
        f1 = self.pfn["l_ank_roll"]
        f1.scale = -self.parameters["swing_scale"]
        self.pfn["l_hip_roll"] = f1

        # f2=mirror f1 in antiphase
        f1.mirror_to(self.afn["l_ank_roll"])
        f1.mirror_to(self.afn["l_hip_roll"])
        # ~ f2=WJFunc()

        f3 = self.pfn["l_hip_pitch"]
        f3.scale = self.parameters["step_scale"]
        f3.offset = self.parameters["step_offset"]

        f33 = self.pfn["l_ank_pitch"]
        f3.mirror_to(f33)
        f33.offset += self.parameters["ankle_offset"]

        f4 = self.pfn["l_knee"]
        f3.mirror_to(f4)
        f4.offset *= 2
        f4.scale *= 2

        f5 = self.afn["l_hip_pitch"]
        f3.copy_to(f5)
        f5.in_scale *= 2
        f5.scale = 0

        f6 = self.afn["l_ank_pitch"]
        f3.mirror_to(f6)
        f6.in_scale *= 2
        f6.scale = f5.scale
        f6.offset += self.parameters["ankle_offset"]

        f7 = self.afn["l_knee"]
        f4.copy_to(f7)
        f7.scale = 0

        self.generate_right()

    def generate_right(self):
        """
        Mirror CPG functions from left to right and antiphase right
        """
        l = [v[2:] for v in self.pfn.keys()]
        for j in l:
            self.afn["l_" + j].mirror_to(self.pfn["r_" + j])
            self.pfn["l_" + j].mirror_to(self.afn["r_" + j])

    def get(self, phase, x, velocity):
        """ Obtain the joint angles for a given phase, position in cycle (x 0,1)) and velocity parameters """
        self.generate()
        angles = {}
        for j in self.pfn.keys():
            if phase:
                angles[j] = self.ang_offsets[j] + self.pfn[j].get(x)
                # print(self.pfn["l_hip_roll"].get(x))
            else:
                angles[j] = self.ang_offsets[j] + self.afn[j].get(x)
        self.apply_velocity(angles, velocity, phase, x)
        return angles

    def show(self):
        """
        Display the CPG functions used
        """
        for j in self.pfn.keys():
            print(j, "p", self.pfn[j], "a", self.afn[j])

    def apply_velocity(self, angles, velocity, phase, x):
        """ Modify on the walk-on-spot joint angles to apply the velocity vector"""

        # VX
        v = velocity[0] * self.parameters["vx_scale"]
        d = (x * 2 - 1) * v
        if phase:
            angles["l_hip_pitch"] += d
            angles["l_ank_pitch"] += d
            angles["r_hip_pitch"] += d
            angles["r_ank_pitch"] += d
        else:
            angles["l_hip_pitch"] -= d
            angles["l_ank_pitch"] -= d
            angles["r_hip_pitch"] -= d
            angles["r_ank_pitch"] -= d

        # VY
        v = velocity[1] * self.parameters["vy_scale"]
        d = (x) * v
        d2 = (1 - x) * v
        if v >= 0:
            if phase:
                angles["l_hip_roll"] -= d
                angles["l_ank_roll"] -= d
                angles["r_hip_roll"] += d
                angles["r_ank_roll"] += d
            else:
                angles["l_hip_roll"] -= d2
                angles["l_ank_roll"] -= d2
                angles["r_hip_roll"] += d2
                angles["r_ank_roll"] += d2
        else:
            if phase:
                angles["l_hip_roll"] += d2
                angles["l_ank_roll"] += d2
                angles["r_hip_roll"] -= d2
                angles["r_ank_roll"] -= d2
            else:
                angles["l_hip_roll"] += d
                angles["l_ank_roll"] += d
                angles["r_hip_roll"] -= d
                angles["r_ank_roll"] -= d

        # VT
        v = velocity[2] * self.parameters["vt_scale"]
        d = (x) * v
        d2 = (1 - x) * v
        if v >= 0:
            if phase:
                angles["l_hip_yaw"] = -d
                angles["r_hip_yaw"] = d
            else:
                angles["l_hip_yaw"] = -d2
                angles["r_hip_yaw"] = d2
        else:
            if phase:
                angles["l_hip_yaw"] = d2
                angles["r_hip_yaw"] = -d2
            else:
                angles["l_hip_yaw"] = d
                angles["r_hip_yaw"] = -d