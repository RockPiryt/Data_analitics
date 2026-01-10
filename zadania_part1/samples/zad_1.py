import numpy as np
from scipy.linalg import hilbert

"""
1. Rozwiąż układ z macierzą Hilberta.
• Niech n = 10. Utwórz macierz Hilberta A ∈ R
n×n
oraz wektor prawej strony b (np.
b = 1 lub wektor z elementami bi = P
j
Aij).
• Rozwiąż układ Ax = b trzema sposobami: numpy.linalg.solve, numpy.linalg.lstsq
oraz przez inv(A).dot(b).
• Porównaj otrzymane rozwiązania, policz błędy (np. norma różnicy względem solve
lub norma residuum |Ax − b| 2) i oblicz cond(A) (np. numpy.linalg.cond).
"""
# --- Tworzenie macierzy Hilberta i prawej strony ---
n = 10
A = hilbert(n)
b = np.ones(n)  # lub: b = A.sum(axis=1)
"""
b = np.ones(n)
Tworzy wektor długości n, w którym wszystkie elementy to 1.
Przykład dla n = 5:[1. 1. 1. 1. 1.]
Używa się go jako prawej strony układu Ax = b.

"""

# --- 1) Rozwiązanie przez numpy.linalg.solve ---
x_solve = np.linalg.solve(A, b)

# --- 2) Rozwiązanie przez najmniejsze kwadraty (lstsq) ---
x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)

# --- 3) Rozwiązanie przez inv(A).dot(b) – metoda NIEZALECANA ---
A_inv = np.linalg.inv(A)
x_inv = A_inv.dot(b)

# --- Błędy i residua ---
res_solve = np.linalg.norm(A @ x_solve - b)
res_lstsq = np.linalg.norm(A @ x_lstsq - b)
res_inv = np.linalg.norm(A @ x_inv - b)

# Różnice względem solve
err_lstsq = np.linalg.norm(x_lstsq - x_solve)
err_inv = np.linalg.norm(x_inv - x_solve)

# --- Warunek macierzy ---
condA = np.linalg.cond(A)

print("Residua |Ax - b|:")
print("solve :", res_solve)
print("lstsq :", res_lstsq)
print("inv   :", res_inv)

print("\nRóżnice względem solve:")
print("||x_lstsq - x_solve|| =", err_lstsq)
print("||x_inv   - x_solve|| =", err_inv)

print("\ncond(A) =", condA)
