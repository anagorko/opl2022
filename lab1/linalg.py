"""
This module, meant for **educational purposes only**, supports learning of basics of linear algebra.

It was created to supplement a "Linear algebra and geometry I" course taught during winter semester 2020
at the University of Warsaw Mathematics Department.

To update, run:

!wget -N https://raw.githubusercontent.com/anagorko/linalg/main/linalg.py

AUTHORS:
    Andrzej Nagórko, Jarosław Wiśniewski

VERSION:
    7/9/2021
"""

from __future__ import annotations

from typing import Dict, List, Optional, Callable, Union
from itertools import combinations
import unittest

import sage.all
import sage.structure.sage_object
import sage.symbolic.expression
import sage.symbolic.operators

from sage.misc.html import HtmlFragment
from sage.repl.ipython_kernel.interact import sage_interactive
from ipywidgets import SelectionSlider

import matplotlib.pyplot as plt
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # We are not running in a notebook


def is_invertible(x):
    """Return true if we can safely divide by x."""

    try:
        return float(x) != 0.0
    except TypeError:
        return False


def format_coefficient(coefficient) -> str:
    """Wrap coefficient in parentheses, if necessary."""

    if isinstance(coefficient, sage.symbolic.expression.Expression):
        if coefficient.operator() == sage.symbolic.operators.add_vararg:
            return f'({sage.all.latex(coefficient)})'
        return sage.all.latex(coefficient)

    if sage.all.latex(coefficient)[0] == '-':
        return f'({sage.all.latex(coefficient)})'

    if hasattr(coefficient, 'coefficients'):
        # Polynomials
        if len(coefficient.coefficients()) > 1:
            return f'({sage.all.latex(coefficient)})'

        return sage.all.latex(coefficient)

    return sage.all.latex(coefficient)


class IMatrix(sage.structure.sage_object.SageObject):
    """Interactive matrix class."""

    def __init__(self, M, separate=0, copy=True, names: Optional[Union[List[str], str]] = None):
        """Instantiate an IMatrix from a matrix-like object M.

        Arguments:
            M - matrix  coefficients.

        Keyword arguments:
            separate - number of columns to separate;
            copy     - make a copy of M, if True;
            var      - names of variables corresponding to non-separated columns.
        """

        self.M: sage.all.Matrix = sage.all.matrix(M) if copy else M
        """Matrix entries."""

        self.separate = separate
        """Column separator placement, counting from the right side."""

        if names is None:
            names = [f'x_{i+1}' for i in range(self.M.ncols() - self.separate)]
        elif isinstance(names, str):
            names = [f'{names}_{i+1}' for i in range(self.M.ncols() - self.separate)]
        else:
            assert len(names) == self.M.ncols() - self.separate

        self.var = sage.all.PolynomialRing(self.M.base_ring(), names=names).gens()
        """Variables corresponding to non-separated columns."""

    def __eq__(self, other):
        return self.M == other.M and self.separate == other.separate and self.var == other.var and \
               self.M.base_ring() == other.M.base_ring()

    def _repr_(self) -> str:
        """Represent IMatrix as a valid Python expression.

        TODO (anagorko): preserve information about coefficient ring of the matrix."""

        return f'IMatrix({repr(list(self.M))}, separate={self.separate}' \
               f', names={[str(v) for v in self.var]})'

    def _latex_(self) -> str:
        """Represent IMatrix as a LaTeX formula."""

        output = list()

        column_format = 'r' * (self.M.ncols() - self.separate) + \
                        ('|' if self.separate > 0 else '') + \
                        'r' * self.separate

        output.append(r'\left[\begin{array}{'f'{column_format}''}')
        for row in self.M:
            output.append(' & '.join([sage.all.latex(el) for el in row]) + r'\\')
        output.append(r'\end{array}\right]')

        return '\n'.join(output)

    def _format_row_operations(self, op: Dict[int, str]) -> str:
        """Represent list operations as a LaTeX formula."""

        output = list()

        operations = [r'\ '] * self.M.nrows()
        for i, operation in op.items():
            operations[i] = operation

        output.append(r'\begin{array}{c}')
        output.append(r'\\'.join(operations))
        output.append(r'\end{array}')

        return '\n'.join(output)

    def _add_multiple_of_row(self, i: int, j: int, s) -> Dict[int, str]:
        """Add s times row j to row i. The operation is done in place."""

        self.M.add_multiple_of_row(i, j, s)

        return {i: fr'+{format_coefficient(s)} \cdot w_{j+1}'}

    def add_multiple_of_row(self, i: int, j: int, s) -> HtmlFragment:
        """Add s times row j to row i. The operation is done in place."""

        return HtmlFragment(''.join([r'\[',
                                     self._latex_(),
                                     self._format_row_operations(self._add_multiple_of_row(i, j, s)),
                                     r'\rightarrow',
                                     self._latex_(),
                                     r'\]']))

    def _swap_rows(self, r1: int, r2: int) -> Dict[int, str]:
        """Swap rows r1 and r2 of self. The operation is done in place."""

        self.M.swap_rows(r1, r2)

        return {r1: fr'\leftarrow r_{r2+1}', r2: fr'\leftarrow r_{r1+1}'}

    def swap_rows(self, r1: int, r2: int) -> HtmlFragment:
        """Swap rows r1 and r2 of self. The operation is done in place."""

        return HtmlFragment(''.join([r'\[',
                                     self._latex_(),
                                     self._format_row_operations(self._swap_rows(r1, r2)),
                                     r'\rightarrow',
                                     self._latex_(),
                                     r'\]']))

    def _lu_swap(self, r1: int, r2: int) -> Dict[int, str]:
        """Swap rows r1 and r2 of self, up to second diagonal.
        Used in LUP factorization. This operation is done in place."""

        for i in range(self.M.nrows() + min(r1, r2)):
            self.M[r1, i], self.M[r2, i] = self.M[r2, i], self.M[r1, i]

        return {r1: fr'\leftarrow r_{r2+1} \setminus 0', r2: fr'\leftarrow r_{r1+1} \setminus 0'}

    def lu_swap(self, r1: int, r2: int) -> HtmlFragment:
        """Swap rows r1 and r2 of self, up to second diagonal.
        Used in LUP factorization. This operation is done in place."""
        return HtmlFragment(''.join([r'\[',
                                     self._latex_(),
                                     self._format_row_operations(self._lu_swap(r1, r2)),
                                     r'\rightarrow',
                                     self._latex_(),
                                     r'\]']))

    def _rescale_row(self, i: int, s) -> Dict[int, str]:
        """Replace i-th row of self by s times i-th row of self. The operation is done in place."""

        self.M.rescale_row(i, s)

        return {i: fr'\cdot {format_coefficient(s)}'}

    def rescale_row(self, i: int, s) -> HtmlFragment:
        """Replace i-th row of self by s times i-th row of self. The operation is done in place."""

        return HtmlFragment(''.join([r'\[',
                                     self._latex_(),
                                     self._format_row_operations(self._rescale_row(i, s)),
                                     r'\rightarrow',
                                     self._latex_() + r'\]']))

    def to_echelon_form(self) -> HtmlFragment:
        """Transform self to the echelon form of self."""

        output = list()

        # Gaussian elimination algorithm is derived from
        # https://ask.sagemath.org/question/8840/how-to-show-the-steps-of-gauss-method/

        col = 0  # all cols before this are already done
        for row in range(0, self.M.nrows()):
            # Need to swap in a nonzero entry from below
            while col < self.M.ncols() and self.M[row][col] == 0:
                for i in self.M.nonzero_positions_in_column(col):
                    if i > row:
                        output.append(r'\[')
                        output.append(self._latex_())
                        output.append(self._format_row_operations(self._swap_rows(row, i)))
                        output.append(r'\rightarrow')
                        output.append(self._latex_())
                        output.append(r'\]')
                        break
                else:
                    col += 1

            if col >= self.M.ncols() - self.separate:
                break

            # Now guaranteed M[row][col] != 0
            if self.M[row][col] != 1:
                if not is_invertible(self.M[row][col]):
         
                    output.append(f'<br>Przerywam eliminację bo nie wiem, czy wyrażenie '
                                  f'${sage.all.latex(self.M[row][col])}$ jest niezerowe.')
                    break
                else:
                    output.append(r'\[')
                    output.append(self._latex_())
                    output.append(self._format_row_operations(self._rescale_row(row, 1 / self.M[row][col])))
                    output.append(r'\rightarrow')
                    output.append(self._latex_())
                    output.append(r'\]')

            change_flag = False
            unchanged = self._latex_()
            operations = dict()
            for changed_row in range(row + 1, self.M.nrows()):
                if self.M[changed_row][col] != 0:
                    change_flag = True
                    factor = -1 * self.M[changed_row][col] / self.M[row][col]
                    operations.update(self._add_multiple_of_row(changed_row, row, factor))

            if change_flag:
                output.append(r'\[')
                output.append(unchanged)
                output.append(self._format_row_operations(operations))
                output.append(r'\rightarrow')
                output.append(self._latex_())
                output.append(r'\]')

            col += 1

        return HtmlFragment('\n'.join(output))

    def to_reduced_form(self):
        """Transform self to the reduced echelon form of self."""
        output = list()

        for changed_row in range(1, self.M.nrows()):
            operations = {i: r'\ ' for i in range(self.M.nrows())}
            unchanged = self._latex_()

            for row in range(0, changed_row):
                factor = -self.M[row][changed_row]
                operations.update(self._add_multiple_of_row(row, changed_row, factor))

            output.append(r'\[')
            output.append(unchanged)
            output.append(self._format_row_operations(operations))
            output.append(r'\rightarrow')
            output.append(self._latex_())
            output.append(r'\]')

        return HtmlFragment('\n'.join(output))

    def as_equations(self) -> SoLE:
        return SoLE(self)

    def as_combination(self) -> LinearCombination:
        return LinearCombination(self)

    def as_determinant(self, coefficient=1) -> Determinant:
        return Determinant(self, coefficient)

    def as_matrix(self) -> IMatrix:
        names = [str(v) for v in self.var]
        return IMatrix(self.M, copy=False, names=names, separate=self.separate)

    def plot(self):
        return self.as_equations().plot()


class SoLE(IMatrix):
    """System of linear equations."""

    def __init__(self, M: IMatrix):
        names = [str(v) for v in M.var]

        super().__init__(M.M, separate=M.separate, copy=False, names=names)

        if self.separate != 1:
            # TODO(anagorko): allow separate=0 and treat it as a homogeneous system
            print('Uwaga: macierz nie wygląda na układ równań.')

        self.x_min = None
        """Plot x range (min)."""
        self.x_max = None
        """Plot x range (max)."""
        self.y_min = None
        """Plot y range (min)."""
        self.y_max = None
        """Plot y range (max)."""

        self.var_sr = [sage.all.var(str(v)) for v in self.var]
        """SR variants of variables."""

    def _format_row_operations(self, op: Dict[int, str]) -> str:
        output = list()

        operations = [r'\ '] * self.M.nrows()
        for i, operation in op.items():
            operations[i] = '/' + operation

        output.append(r'\begin{array}{c}')
        output.append(r'\\'.join(operations))
        output.append(r'\end{array}')

        return '\n'.join(output)

    def _latex_(self) -> str:
        output = list()

        column_format = 'c' * (self.M.ncols() - self.separate) * 2 + 'l' * self.separate

        output.append(r'\left\{\begin{array}{'f'{column_format}''}')
        output += ['&'.join(self._format_row(row)) + r'\\' for row in self.M]
        output.append(r'\end{array}\right.')

        return '\n'.join(output)

    def _format_row(self, row: List) -> List[str]:
        """A latex representation of a row."""

        empty_so_far = True
        row_output = list()
        for i, coefficient in enumerate(row):
            sign = ''
            if i > 0 and not empty_so_far:
                if i == self.M.ncols() - self.separate:
                    sign = '='
                elif i < self.M.ncols() - self.separate:
                    if coefficient == 0.0:
                        sign = ''
                    elif coefficient < 0.0:
                        sign = '-'
                        coefficient = -coefficient
                    else:
                        sign = '+'

            variable = 1
            if coefficient != 0.0 and i < self.M.ncols() - self.separate:
                variable = self.var[i]
            term = variable * coefficient

            if term == 0.0 and i < self.M.ncols() - self.separate:
                term = ''

            if i == 0:
                row_output.append(sign + sage.all.latex(term))
            else:
                row_output.append(sign)
                row_output.append(sage.all.latex(term))

            if coefficient != 0.0:
                empty_so_far = False

        return row_output

    """
    Methods related to plot() function.
    """

    MAX_ASPECT_RATIO = 2.0
    SLIDER_RANGE = [sage.all.QQ(i - 125) / 50 for i in range(251)]

    @staticmethod
    def _row_to_function(coefficients: List) -> Callable:
        """Converts a list of coefficients into a linear function."""

        def f(*args):
            return (1 / coefficients[-2]) * (coefficients[-1] - sum(v * coefficients[i] for i, v in enumerate(args)))

        return f

    def _equation(self, coefficients: List) -> sage.symbolic.expression.Expression:
        return sum(self.var_sr[i] * coefficients[i] for i in range(len(coefficients) - 1)) == coefficients[-1]

    def _format_equation(self, row: List) -> str:
        return f'${"".join(self._format_row(row))}$'

    def plot(self):
        """Interactive plot of self. Supported in two dimensions (so far)."""

        if self.M[0, 0].parent() == sage.all.SR:
            free_variables = list(sum(sum(self.M)).free_variables())
        else:
            free_variables = list()

        var_sliders = {str(var): SelectionSlider(options=SoLE.SLIDER_RANGE, continuous_update=False, value=0)
                       for var in free_variables}

        def f(**_var_sliders):
            M = self.M.subs({sage.all.var(v): _var_sliders[v] for v in _var_sliders})

            avg_y = 0
            """Average y-value of subsystems consisting of two equations."""

            """Determine x- and y- range of the plot."""
            for i, j in combinations(range(M.nrows()), 2):
                solution = sage.all.solve([self._equation(M[i]), self._equation(M[j])],
                                          *self.var_sr, solution_dict=True)
                if solution:
                    solution = solution[0]
                    if solution[self.var_sr[0]].free_variables() or solution[self.var_sr[1]].free_variables():
                        continue

                    avg_y += solution[self.var_sr[1]]

                    if self.x_min is None or solution[self.var_sr[0]] - 1 < self.x_min:
                        self.x_min = solution[self.var_sr[0]] - 1

                    if self.x_max is None or solution[self.var_sr[0]] + 1 > self.x_max:
                        self.x_max = solution[self.var_sr[0]] + 1

            if self.x_min is None:
                self.x_min = -10.0

            if self.x_max is None:
                self.x_max = 10.0

            avg_y = avg_y / M.nrows()
            self.y_max = avg_y + SoLE.MAX_ASPECT_RATIO * (self.x_max - self.x_min) / 2
            self.y_min = avg_y - SoLE.MAX_ASPECT_RATIO * (self.x_max - self.x_min) / 2

            x = np.arange(self.x_min, self.x_max, sage.all.QQ((self.x_max - self.x_min)/100))

            fig, ax = plt.subplots()

            for i in range(M.nrows()):
                if M[i][-2] == 0.0:
                    # vertical line
                    if M[i][0] != 0.0:
                        xv = M[i][-1] / M[i][0]
                        ax.plot([xv, xv], [self.y_min, self.y_max], label=self._format_equation(M[i]))
                else:
                    row_f = SoLE._row_to_function(M[i])
                    y = [row_f(_x) for _x in x]

                    x_clip, y_clip = list(), list()

                    for _x, _y in zip(x, y):
                        if self.y_min <= _y <= self.y_max:
                            x_clip.append(_x)
                            y_clip.append(_y)

                    ax.plot(x_clip, y_clip, label=self._format_equation(M[i]))

            ax.set(xlabel=self.var[0], ylabel=self.var[1])
            ax.grid()

            plt.legend()
            plt.show()

        w = sage_interactive(f, **var_sliders)
        # output = w.children[-1]
        # output.layout.height = '600px'

        default_dpi = plt.rcParamsDefault['figure.dpi']
        plt.rcParams['figure.dpi'] = default_dpi * 1.5

        display(w)

    def _repr_(self) -> str:
        return f'IMatrix({repr(list(self.M))}, separate={self.separate}).as_equations()'


class LinearCombination(IMatrix):
    """System of linear equations interpreted as linear combination."""

    def __init__(self, M: IMatrix):
        names = [str(v) for v in M.var]
        super().__init__(M.M, separate=M.separate, copy=False, names=names)

        if self.separate != 1:
            print('Uwaga: macierz nie wygląda na układ równań.')

    def _format_column(self, col_n: int) -> str:
        """Format column as a column vector."""

        output = list()

        output.append(r'\left[\begin{array}{c}')
        output += [sage.all.latex(self.M[i][col_n]) + r'\\' for i in range(self.M.nrows())]
        output.append(r'\end{array}\right]')

        return '\n'.join(output)

    def _latex_(self) -> str:
        lhs = list()
        for i in range(self.M.ncols() - self.separate):
            lhs.append(str(self.var[i]) + self._format_column(i))

        output = ['+'.join(lhs), '=', self._format_column(-1)]

        return ' '.join(output)

    def _repr_(self) -> str:
        return f'IMatrix({repr(list(self.M))}, separate={self.separate}).as_combination()'


class Determinant(IMatrix):
    """A determinant of a square matrix, with coefficient."""

    def __init__(self, M: IMatrix, coefficient=1):
        names = [str(v) for v in M.var]
        super().__init__(M.M, separate=M.separate, copy=False, names=names)

        assert self.M.nrows() == self.M.ncols(), "Macierz wyznacznika musi być kwadratowa."
        assert self.separate == 0, "Macierz wyznacznika nie może mieć wyrazów wolnych."

        self.coefficient = coefficient

    def _repr_(self) -> str:
        return f'IMatrix({repr(list(self.M))}, separate={self.separate}).as_determinant({self.coefficient})'

    def _latex_(self) -> str:
        """Represent Determinant as a LaTeX formula."""

        output = list()

        column_format = 'r' * (self.M.ncols() - self.separate) + \
                        ('|' if self.separate > 0 else '') + \
                        'r' * self.separate

        if self.coefficient != 1:
            output.append('(' + str(self.coefficient) + r')\cdot')

        output.append(r'\left|\begin{array}{'f'{column_format}''}')
        for row in self.M:
            output.append(' & '.join([sage.all.latex(el) for el in row]) + r'\\')
        output.append(r'\end{array}\right|')

        return '\n'.join(output)

    def _swap_rows(self, r1: int, r2: int) -> Dict[int, str]:
        if r1 != r2:
            self.coefficient = -self.coefficient

        return super()._swap_rows(r1, r2)

    def _rescale_row(self, i: int, s) -> Dict[int, str]:
        assert s != 0, "Mnożenie wiersza przez zero nie jest odwracalne"

        self.coefficient = (1/s) * self.coefficient

        return super()._rescale_row(i, s)

    def row_expansion(self, i: int) -> HtmlFragment:
        """Laplace expansion on i-th row."""

        output = list()
        output.append(r'\[')
        for j in range(self.M.ncols()):
            if j > 0:
                output.append('+')

            output.append(r'(-1)^{' f'{i+1} + {j+1}' r'}\cdot')
            output.append(f'({self.M[i, j]})'
                          r'\cdot')

            N = self.M[[k for k in range(self.M.nrows()) if k != i],
                       [k for k in range(self.M.ncols()) if k != j]]
            output.append(IMatrix(N).as_determinant()._latex_())
        output.append(r'\]')

        return HtmlFragment(''.join(output))

    def col_expansion(self, i: int) -> HtmlFragment:
        """Laplace expansion on i-th column."""

        output = list()
        output.append(r'\[')
        for j in range(self.M.nrows()):
            if j > 0:
                output.append('+')

            output.append(r'(-1)^{' f'{j+1} + {i+1}' r'}\cdot')
            output.append(f'({self.M[j, i]})'
                          r'\cdot')

            N = self.M[[k for k in range(self.M.nrows()) if k != j],
                       [k for k in range(self.M.ncols()) if k != i]]
            output.append(IMatrix(N).as_determinant()._latex_())
        output.append(r'\]')

        return HtmlFragment(''.join(output))


class IMatrixTest(unittest.TestCase):
    def test_serialization(self):
        """Test repr/eval."""

        M = IMatrix([[1, 2, 3], [4, 5, 6]], separate=1, names='y')
        self.assertEqual(M, eval(repr(M)))
        # N = IMatrix(sage.all.matrix(sage.all.GF(11), [[3, 2, 1], [6, 5, 4]]), separate=1, names='y')
        # self.assertEqual(N, eval(repr(N)))


def main():
    """Basic functionality check, ran when script is invoked from the command line."""

    M = IMatrix([[1, 2, 3], [4, 5, 6]])
    print(M.add_multiple_of_row(0, 1, 3))


if __name__ == '__main__':
    main()
else:
    print(__doc__)
    from IPython.core.display import HTML
    try:
        display(HTML("<style>.container { width:99% !important; }</style>"))
    except NameError:
        pass  # We are not running in a notebook
