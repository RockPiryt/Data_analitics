import numpy as np
from scipy.linalg import hilbert

"""
1. Rozwiąż układ z macierzą Hilberta.
• Niech n = 10. Utwórz macierz Hilberta A ∈ R n×n oraz wektor prawej strony b (np.b = 1 lub wektor z elementami bi = PjAij).
• Rozwiąż układ Ax = b trzema sposobami: numpy.linalg.solve, numpy.linalg.lstsq
oraz przez inv(A).dot(b).
• Porównaj otrzymane rozwiązania, policz błędy (np. norma różnicy względem solve
lub norma residuum |Ax − b| 2) i oblicz cond(A) (np. numpy.linalg.cond).
"""
# Tworzenie macierzy Hilberta
n = 10
A = hilbert(n)

# Utworzenie wektora prawej strony
"""
Liczy sumę każdego wiersza macierzy A i zapisuje do wektora b.
axis=1 → sumowanie po kolumnach (czyli po wierszu).
"""
b = A.sum(axis=1)


x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
