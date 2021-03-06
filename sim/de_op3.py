# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import Task
from core.op3model import Griewank
from niapy.algorithms.basic import DifferentialEvolution

# we will run Differential Evolution for 5 independent runs
algo = DifferentialEvolution(population_size=5, differential_weight=0.5, crossover_probability=0.9)
for i in range(5):
    task = Task(problem=Griewank(dimension=6), max_iters=40)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())