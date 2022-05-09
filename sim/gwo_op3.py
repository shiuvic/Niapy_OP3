# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import GreyWolfOptimizer
from niapy.task import Task
from core.op3model import Sphere

# we will run Grey Wolf Optimizer for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=5), max_iters=40)
    algo = GreyWolfOptimizer(population_size=5)
    best = algo.run(task)
    print(best)