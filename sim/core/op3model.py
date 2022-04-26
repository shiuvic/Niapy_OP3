# encoding=utf8

"""Implementation of Griewank funcion."""

import numpy as np
from niapy.problems.problem import Problem
from core.run import run
__all__ = ['Griewank', 'ExpandedGriewankPlusRosenbrock']


class Griewank(Problem):

    def __init__(self, dimension=4, lower=-0.6, upper=0.6, *args, **kwargs):
        r"""Initialize Griewank problem..

        Args:
            dimension (Optional[int]): Dimension of the problem.
            lower (Optional[Union[float, Iterable[float]]]): Lower bound of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bound of the problem.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        super().__init__(dimension, lower, upper, *args, **kwargs)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D \frac{x_i^2}{4000} - \prod_{i=1}^D \cos(\frac{x_i}{\sqrt{i}}) + 1$'''

    def _evaluate(self, sol):
        val = 0.0
        for i in range(self.dimension):
            val = sol[i]

        parameters = {"swing_scale": 0.0,
                      "step_scale": sol[1],
                      "step_offset": sol[2],
                      "ankle_offset": 0.0,
                      "vx_scale": sol[3],
                      "vy_scale": sol[4],
                      "vt_scale": sol[5]}
        walk_offset = {'hip_pitch': -0.063,
                       'hip_roll': 0.0,
                       'hip_yaw': 0.0,
                       'ank_pitch': 0.0,
                       'ank_roll': 0.0,
                       'knee': 0.0}

        for key, value in parameters.items():
            print('{key}:{value}'.format(key=key, value=value), flush=True)
        for key, value in walk_offset.items():
            print('{key}:{value}'.format(key=key, value=value), flush=True)

        x = run(1, 0, 0, parameters, walk_offset)
        val = (0 - np.linalg.norm(x - [0.0, 0.0, 0.0]))
        print("----val------->", val, flush=True)
        return val

class Sphere(Problem):

    def __init__(self, dimension=4, lower=-0.6, upper=0.6, *args, **kwargs):
        r"""Initialize Sphere problem..

        Args:
            dimension (Optional[int]): Dimension of the problem.
            lower (Optional[Union[float, Iterable[float]]]): Lower bounds of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bounds of the problem.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        super().__init__(dimension, lower, upper, *args, **kwargs)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D x_i^2$'''

    def _evaluate(self, sol):
        val = 0.0
        for i in range(self.dimension):
            val = sol[i]

        parameters = {"swing_scale": 0.0,
                      "step_scale": sol[1],
                      "step_offset": sol[2],
                      "ankle_offset": 0.0,
                      "vx_scale": sol[3],
                      "vy_scale": sol[4],
                      "vt_scale": sol[5]}
        walk_offset = {'hip_pitch': -0.063,
                       'hip_roll': 0.0,
                       'hip_yaw': 0.0,
                       'ank_pitch': 0.0,
                       'ank_roll': 0.0,
                       'knee': 0.0}

        for key, value in parameters.items():
            print('{key}:{value}'.format(key=key, value=value), flush=True)
        for key, value in walk_offset.items():
            print('{key}:{value}'.format(key=key, value=value), flush=True)

        x = run(1, 0, 0, parameters, walk_offset)
        val = (0 - np.linalg.norm(x - [0.0, 0.0, 0.0]))
        print("----val------->", val, flush=True)
        return val

class ExpandedGriewankPlusRosenbrock(Problem):
    r"""Implementation of Expanded Griewank's plus Rosenbrock function.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function: **Expanded Griewank's plus Rosenbrock function**

        :math:`f(\textbf{x}) = h(g(x_D, x_1)) + \sum_{i=2}^D h(g(x_{i - 1}, x_i)) \\ g(x, y) = 100 (x^2 - y)^2 + (x - 1)^2 \\ h(z) = \frac{z^2}{4000} - \cos \left( \frac{z}{\sqrt{1}} \right) + 1`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = h(g(x_D, x_1)) + \sum_{i=2}^D h(g(x_{i - 1}, x_i)) \\ g(x, y) = 100 (x^2 - y)^2 + (x - 1)^2 \\ h(z) = \frac{z^2}{4000} - \cos \left( \frac{z}{\sqrt{1}} \right) + 1$

        Equation:
            \begin{equation} f(\textbf{x}) = h(g(x_D, x_1)) + \sum_{i=2}^D h(g(x_{i - 1}, x_i)) \\ g(x, y) = 100 (x^2 - y)^2 + (x - 1)^2 \\ h(z) = \frac{z^2}{4000} - \cos \left( \frac{z}{\sqrt{1}} \right) + 1 \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference:
        http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf

    """

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize Expanded Griewank's plus Rosenbrock problem..

        Args:
            dimension (Optional[int]): Dimension of the problem.
            lower (Optional[Union[float, Iterable[float]]]): Lower bounds of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bounds of the problem.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        super().__init__(dimension, lower, upper, *args, **kwargs)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\textbf{x}) = h(g(x_D, x_1)) + \sum_{i=2}^D h(g(x_{i - 1}, x_i)) \\ g(x, y) = 100 (x^2 - y)^2 + (x - 1)^2 \\ h(z) = \frac{z^2}{4000} - \cos \left( \frac{z}{\sqrt{1}} \right) + 1$'''

    def _evaluate(self, x):
        x1 = 100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0
        x2 = x1 * x1 / 4000.0 - np.cos(x1 / np.sqrt(np.arange(1, self.dimension)))
        return np.sum(x2)
