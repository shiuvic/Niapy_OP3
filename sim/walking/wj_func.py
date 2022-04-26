import math


class WJFunc:
    """
    Walk Joint Function CPG style
    Provides parameterized sine wave functions as y=offset+scale*(in_offset+in_scale*x)
    """

    def __init__(self):
        self.offset = 0
        self.scale = 1
        self.in_offset = 0
        self.in_scale = 1

    def get(self, x):
        """ x between 0 and 1"""
        f = math.sin(self.in_offset + self.in_scale * x)
        return self.offset + self.scale * f

    def clone(self):
        z = WJFunc()
        z.offset = self.offset
        z.scale = self.scale
        z.in_offset = self.in_offset
        z.in_scale = self.in_scale
        return z

    def mirror(self):
        z = self.clone()
        z.offset *= -1
        z.scale *= -1
        return z

    def copy_to(self, wj):
        wj.offset = self.offset
        wj.scale = self.scale
        wj.in_offset = self.in_offset
        wj.in_scale = self.in_scale

    def mirror_to(self, wj):
        self.copy_to(wj)
        wj.offset *= -1
        wj.scale *= -1

    def __str__(self):
        return "y=%f+%f*sin(%f+%f*x)" % (self.offset, self.scale, self.in_offset, self.in_scale)