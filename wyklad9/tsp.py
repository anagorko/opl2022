import math
from copy import copy
from scipy.stats import uniform
from numpy.random import seed as numpy_seed
from scipy.spatial import distance
import networkx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

from sage.all import scatter_plot, circle, text, InteractiveLPProblemStandardForm, matrix, vector, RR, QQ, arrow, show


epsilon = 10**-6
R = RR


class Coords:
    def __init__(self, N = 10, seed = None):
        """N random cities on a unit disc."""

        if seed:
            numpy_seed(seed)

        r = uniform.rvs(scale=1, size=N)
        theta = uniform.rvs(scale=2*math.pi, size=N)
        norm = max(r)

        self.coords = [(r[i]/norm*math.cos(theta[i]), r[i]/norm*math.sin(theta[i])) for i in range(N)]

    def plot(self):
        """Plot of cities."""

        return circle((0, 0), 1.0) + scatter_plot(self.coords, markersize=200, zorder=6, axes=False) + sum([text(i, c, zorder=7) for i, c in enumerate(self.coords)])

    def linear_problem(self) -> InteractiveLPProblemStandardForm:
        """LP representation of TSP, with multiple cycles allowed."""

        var_names = [f'x_{i}{j}' for i, a in enumerate(self.coords) for j, b in enumerate(self.coords) if a != b]
        edges = [(a, b) for a in self.coords for b in self.coords if a != b]

        A = matrix(R, 0, len(edges))
        b = matrix(R, 0, 1)
        c = vector(R, [-distance.euclidean(a,b) for a in self.coords for b in self.coords if a != b])

        # z każdego wierzchołka wychodzi dokładnie jedna krawędź
        for a in self.coords:
            row = vector(R, len(edges))
            for i, (d, _) in enumerate(edges):
                if d == a:
                    row[i] = 1
            A = A.stack(row)
            b = b.stack(vector(R, [1]))
            A = A.stack(-row)
            b = b.stack(vector(R, [-1]))

        # do każdego wierzchołka wchodzi dokładnie jedna krawędź
        for a in self.coords:
            row = vector(R, len(edges))
            for i, (_, d) in enumerate(edges):
                if d == a:
                    row[i] = 1
            A = A.stack(row)
            b = b.stack(vector(R, [1]))

        # Wartości w przedziale [0,1]

        for i, (u, v) in enumerate(edges):
            row = vector(R, len(edges))
            row[i] = 1
            A = A.stack(row)
            b = b.stack(vector(R, [1]))

        return InteractiveLPProblemStandardForm(A, b, c, x=var_names, slack_variables='y')

    def plot_solution(self, P):
        """Draw a solution of TSP LP."""

        assert is_binary(P)

        x = [P.x()[i] for i, v in enumerate(P.optimal_solution()) if v == 1.0]

        edges = self.plot()

        for v in x:
            edges = edges + arrow(self.coords[int(str(v)[2])], self.coords[int(str(v)[3])])

        return edges

    def print_solution(self, P):
        """Print a solution of TSP LP."""

        print_solution(P)


def print_solution(P):
    """Print a solution of TSP LP."""

    x = [P.x()[i] for i, v in enumerate(P.optimal_solution()) if v == 1.0]
    for i, v in enumerate(P.optimal_solution()):
        if v == 0.0:
            continue

        src = str(P.x()[i])[2]
        dst = str(P.x()[i])[3]

        print(f'{src} --> {dst}: {v}')

    print(problem_description(P))


def x(P, src, dst):
    """Returns Goedel number of an edge, i.e. column index of an edge in matrix A."""

    var_names = list(map(str, P.x()))
    return var_names.index(f'x_{src}{dst}')


def remove_subtour(P, subtour):
    """Adds constraint to P to eliminate specified subtour."""

    subtour = copy(subtour)

    row = vector(R, len(P.x()))

    subtour.append(subtour[0])

    for i in range(len(subtour) - 1):
        src = subtour[i]
        dst = subtour[i+1]
        row[x(P, src, dst)] = 1.0
        # row[x(P, dst, src)] = 1.0

    A, b, c, _x = P.Abcx()

    A1 = A.stack(row)
    b1 = list(b)
    b1.append(len(subtour) - 2)
    b1 = vector(R, b1)
    P1 = InteractiveLPProblemStandardForm(A1, b1, c, x=_x, slack_variables='y')

    return P1


def is_binary(P):
    """Check if all variables in the solution are binary."""

    return all([x == 0.0 or x == 1.0 for x in P.optimal_solution()])


def find_subtours(P):
    """Find all subtours, assuming the solution is binary."""

    assert is_binary(P)

    N = int((1 + math.sqrt(1 + 4*len(P.x()))) / 2)

    cities = set(range(N))

    tours = []

    while cities:
        starting_city = cities.pop()
        tour = [starting_city]
        city = starting_city
        next_city = None

        while next_city != starting_city:
            next_city = None
            for i in range(N):
                if i == city:
                    continue
                if P.optimal_solution()[x(P, city, i)] == 1.0:
                    next_city = i
                    break
            assert next_city is not None
            if next_city != starting_city:
                tour.append(next_city)
                cities.remove(next_city)
            city = next_city
        tours.append(tour)

    return tours


def branch_on(P, src, dst):
    """Branch on a fractional variable x_{src}{dst}."""

    A, b, c, _x = P.Abcx()

    # Problem P0 has x_{src}{dst} set to 0
    r0  = vector(R, len(P.x()))
    r0[x(P, src, dst)] = 1.0
    A0 = A.stack(r0)
    b0 = list(b)
    b0.append(0)
    b0 = vector(R, b0)
    P0 = InteractiveLPProblemStandardForm(A0, b0, c, x=_x, slack_variables='y')

    # Problem P1 has x_{src}{dst} set to 1
    r1  = vector(R, len(P.x()))
    r1[x(P, src, dst)] = -1.0
    A1 = A.stack(r1)
    b1 = list(b)
    b1.append(-1)
    b1 = vector(R, b1)
    P1 = InteractiveLPProblemStandardForm(A1, b1, c, x=_x, slack_variables='y')

    return P0, P1


def gomory_cut(P, v):
    I = P.initial_dictionary()
    F = P.final_dictionary()
    var_names = list(map(str, P.x()))
    row = vector(R, len(P.x()))
    row[var_names.index(v)] = 1.0
    coeffs = F.row_coefficients(v)
    rhs = floor(F.constant_terms()[list(map(str, F.basic_variables())).index(v)])
    for i, x in enumerate(F.nonbasic_variables()):
        c = coeffs[i]
        if x in I.basic_variables():
            if c < 0:
                row += -floor(c) * I.row_coefficients(x)
                rhs += I.constant_terms()[list(map(str, I.basic_variables())).index(str(x))]
        else:
            if c < 0:
                row[var_names.index(x)] = floor(c)
    A, b, c, x = P.Abcx()
    A1 = A.stack(row)
    b1 = list(b)
    b1.append(rhs)
    b1 = vector(R, b1)
    P1 = InteractiveLPProblemStandardForm(A1, b1, c, x=x, slack_variables='y')
    return P1


def problem_description(P, verbose=True):
    if not P.is_feasible():
        return 'INFEASIBLE' if verbose else 'INFEAS'
    if is_binary(P):
        if len(find_subtours(P)) > 1:
            return 'BINARY MULTIPLE SUBTOURS' if verbose else f'MULT {float(-P.optimal_value()):.3f}'
        else:
            return f'BINARY FEASIBLE {-P.optimal_value()}' if verbose else f'FEAS {float(-P.optimal_value()):.3f}'
    else:
        return f'FRACTIONAL {-P.optimal_value()}' if verbose else f'FRAC {float(-P.optimal_value()):.3f}'


class Tree:
    def __init__(self, coords):
        self.coords = coords
        P = coords.linear_problem()
        self.G = networkx.DiGraph()
        self.G.add_node(0)
        self.G.nodes[0]['label'] = f'0: {problem_description(P, False)}'
        self.G.nodes[0]['problem'] = P

    def plot(self):
        plt.figure(figsize=(20,10))
        pos=graphviz_layout(self.G, prog='dot')
        networkx.draw_networkx(self.G, labels=networkx.get_node_attributes(self.G, 'label'), pos=pos, node_size=500, node_color='#eeeeee', edge_labels=networkx.get_edge_attributes(self.G, 'label'))
        networkx.draw_networkx_edge_labels(self.G, pos=pos, edge_labels=networkx.get_edge_attributes(self.G, 'label'))
        plt.show()

    def remove_subtour(self, v, tour):
        n = self.G.number_of_nodes()
        self.G.add_node(n)

        P = self.G.nodes[v]['problem']
        Pn = remove_subtour(P, tour)
        Pn = remove_subtour(Pn, list(reversed(tour)))
        self.G.nodes[n]['label'] = f'{n}: {problem_description(Pn, False)}'
        self.G.nodes[n]['problem'] = Pn

        self.G.add_edge(n, v)
        self.G.edges[n, v]['label'] = f'RM {tour}'

    def print_subtours(self, v):
        print(find_subtours(self.G.nodes[v]['problem']))

    def branch_on(self, v, edge):
        src = edge // 10
        dst = edge % 10

        n0 = self.G.number_of_nodes()
        n1 = n0 + 1

        self.G.add_node(n0)
        self.G.add_node(n1)

        P0, P1 = branch_on(self.G.nodes[v]['problem'], src, dst)

        self.G.nodes[n0]['label'] = f'{n0}: {problem_description(P0, False)}'
        self.G.nodes[n0]['problem'] = P0
        self.G.nodes[n1]['label'] = f'{n1}: {problem_description(P1, False)}'
        self.G.nodes[n1]['problem'] = P1

        self.G.add_edge(n0, v)
        self.G.add_edge(n1, v)

        self.G.edges[n0, v]['label'] = f'x_{src}{dst} = 0'
        self.G.edges[n1, v]['label'] = f'x_{src}{dst} = 1'

    def print(self, v):
        print_solution(self.G.nodes[v]['problem'])

    def solution(self, v):
        show(self.coords.plot_solution(self.G.nodes[v]['problem']))

    def _gomory_cut(self, P, u):
        """Gomory cut on problem P, variable u"""
        I = P.initial_dictionary()
        F = P.final_dictionary()
        var_names = list(map(str, P.x()))
        row = vector(R, len(P.x()))
        row[var_names.index(u)] = 1.0
        coeffs = F.row_coefficients(u)
        rhs = math.floor(F.constant_terms()[list(map(str, F.basic_variables())).index(u)])
        for i, x in enumerate(F.nonbasic_variables()):
            c = coeffs[i]
            if x in I.basic_variables():
                if c < 0:
                    row += -math.floor(c) * I.row_coefficients(x)
                    rhs += I.constant_terms()[list(map(str, I.basic_variables())).index(str(x))]
            else:
                if c < 0:
                    row[var_names.index(str(x))] = math.floor(c)
        A, b, c, x = P.Abcx()
        A1 = A.stack(row)
        b1 = list(b)
        b1.append(rhs)
        b1 = vector(R, b1)
        P1 = InteractiveLPProblemStandardForm(A1, b1, c, x=x, slack_variables='y')
        return P1

    def gomory_cut(self, v):
        """Gomory cut on first fractional variable of the original problem."""

        P = self.G.nodes[v]['problem']
        P1 = P
        for variable, value in zip(P.x(), P.optimal_solution()):
            if value > epsilon and value < 1.0 - epsilon:
                print(f'Cutting on {variable} value {value}', P1.final_dictionary().basic_variables())
                P1 = self._gomory_cut(P1, str(variable))
                break

        n1 = self.G.number_of_nodes()
        self.G.add_node(n1)
        self.G.nodes[n1]['problem'] = P1
        self.G.nodes[n1]['label'] = f'{n1}: {problem_description(P1, False)}'

        self.G.add_edge(n1, v)
        self.G.edges[n1, v]['label'] = 'gomory'
