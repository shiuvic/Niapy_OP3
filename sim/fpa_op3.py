# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import FlowerPollinationAlgorithm
from niapy.task import Task
from core.op3model import Sphere

# we will run Flower Pollination Algorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=6), max_iters=40)
    algo = FlowerPollinationAlgorithm(population_size=5, p=0.5)
    best = algo.run(task=task)
    print(best)