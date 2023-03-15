from sage.all import * 
from itertools import combinations

def zbiory_bazowe_dopuszczalne(A, b):
    """Zwraca zbiór zbiorów bazowych dopuszczalnych dla problemu w postaci równościowej Ax = b, x >= 0."""
    
    n = A.ncols()  # wymiar przestrzeni
    m = A.nrows()  # liczba równań

    bazowe_dopuszczalne = set()  # zbiór
    
    for Base in combinations(range(n), m):
        A_Base = A.matrix_from_columns(Base)
        if A_Base.det() != 0:
            if  all(i >= 0 for i in ~A_Base * b):
                bazowe_dopuszczalne.add(Base)

    return bazowe_dopuszczalne


def rozwiazanie_bazowe(A, b, Base):
    """Zwraca rozwiązanie bazowe dla zbioru bazowego Base i problemu w postaci równościowej Ax = b, x >=0 ."""
    
    A_Base = A.matrix_from_columns(Base)
    if A_Base.det() == 0:
        raise ValueError(f'Zbiór {Base} nie jest bazowy.')
        
    x_Base = dict(zip(Base, ~A_Base * b))  # przyporządkowanie i -> x_i dla i \in Base    

    return tuple(x_Base.get(i, 0) for i in range(A.ncols()))  # x_i dla i \in Base, 0 dla pozostałych
