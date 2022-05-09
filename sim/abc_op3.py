# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import ArtificialBeeColonyAlgorithm
from niapy.task import Task
from core.op3model import Sphere

# we will run Artificial Bee Colony Algorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=5), max_iters=40)
    algo = ArtificialBeeColonyAlgorithm(population_size=5, limit=2)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))