from walker import Walker
import time
import math
walker = Walker(fallen_reset=True)
walker.reset_and_start()
def run(x_vel, y_vel, ang_vel,parameters,walk_offset):
    walker.update_new_vel(x_vel, y_vel, ang_vel,parameters,walk_offset)
    time.sleep(1)

    while True:
        if walker.fall:
            return (walker.robot_postion())
        time.sleep(0.001)
