from sage.all import * 
from itertools import combinations

from opl import zbiory_bazowe_dopuszczalne, rozwiazanie_bazowe


def krawedz(a, b):
    """Rysuje krawędź od punktu a do punktu b. Punkty muszą być w R^2 lub R^3."""
    
    return line((a, b), thickness=2) + point(a) + point(b)

def promien(a, v):
    """Rysuje promień od punktu a w kierunku v. Punkt i promień muszą być w R^2 lub R^3."""
    a = vector(a)
    v = vector(v)
    
    if len(a) == 2:
        return point(a) + line((a, a + v), thickness=2) + line((a + v, a + 3/2 * v), thickness=2, linestyle='dashed')
    else:
        return point(a) + line((a, a + v), thickness=2) + line((a + v, a + 2* v), thickness=2, opacity=0.3)

def rysuj(krawedzie, promienie):
    rysunek = Graphics()
    
    for a, b in krawedzie:
        rysunek += krawedz(a, b)
    for a, v in promienie:
        rysunek += promien(a, v)

    return rysunek





def krawedzie_promienie(A, b, Base):
    """Funkcja wyznacza wszystkie krawędzie i promienie wychodzące ze wierzchołka bazowego
    odpowiadającego zbiorowiu bazowemu Base.
    
    Zakładamy, że problem jest dany w postaci równościowej:  Ax = b, x >= 0."""
    
    n = A.ncols()  # wymiar przestrzeni
    m = A.nrows()  # liczba nierówności

    krawedzie = set()
    promienie = set()
    
    A_Base = A.matrix_from_columns(Base)
    x = vector(rozwiazanie_bazowe(A, b, Base))
    
    for i in range(n):
        if i not in Base:           
            delta_Base = dict(zip(Base, ((~A_Base) * A).matrix_from_columns((i, ))))
            delta = vector(delta_Base.get(i, (0, ))[0] for i in range(n))

            # dlugość kroku
            t = float('inf')
            for x_i, delta_i in zip(x, delta):
                if delta_i > 0:
                    t = min(t, x_i / delta_i)
            
            delta[i] = -1
            
            if t == float('inf'):
                # promień
                delta = -1 * delta

                promienie.add((tuple(x), tuple(delta)))
            else:
                # krawędź
                delta = -t * delta

                krawedzie.add((tuple(x), tuple(x + delta)))

    return krawedzie, promienie

def rysuj_Abp(A, b, p):
    """Zwraca wizualizację szkieletu dla problemu w postaci równościowej Ax = b, x >= 0 i rzutowania p."""

    szkielet = Graphics()

    # Wyznaczamy rozwiazania bazowe dopuszczalne
    for Base in zbiory_bazowe_dopuszczalne(A, b):
        krawedzie, promienie = krawedzie_promienie(A, b, Base)
        
        for u, v in krawedzie:
            szkielet += krawedz(p(u), p(v))
        for u, v in promienie:
            szkielet += promien(p(u), p(v))

    return szkielet

from sage.plot.plot3d.plot3d import axes

def rysuj_nier(A, b, c):
    P = InteractiveLPProblemStandardForm(A, b, c)
    show(P)
    n = A.ncols()
    A = A.augment(identity_matrix(A.nrows()))

    def rzut(p):
        return p[0:n]

    if n == 3:
        if c.norm() != 0:
            show(rysuj_Abp(A, b, rzut) + line(((0,0,0), c), thickness=6, arrow_head=True, color='red') + axes(radius=1, color='black'), frame=False, aspect_ratio=1)
        else:
            show(rysuj_Abp(A, b, rzut) + axes(radius=1, color='black'), frame=False, aspect_ratio=1)
    else:
        if c.norm() != 0:
            show(rysuj_Abp(A, b, rzut) + arrow((0, 0), c.normalized(), thickness=6, color='red'), aspect_ratio=1)
        else:
            show(rysuj_Abp(A, b, rzut), aspect_ratio=1)
            
