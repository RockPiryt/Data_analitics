import numpy as np
import numpy as np
from scipy.linalg import hilbert
import matplotlib.pyplot  as plt

n = 10
A = hilbert(n)

# Utworzenie wektora prawej strony, sumowanie po kolumnach
b = A.sum(axis=1)

# rozkład LU - wyniki najdokładniejsze
x_solve = np.linalg.solve(A, b)

# metoda najmniejsych kwadratów - wyniki średnie
x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)

# rozwiazywanie układu przez mnożenie odwrotnoscią
A_inv = np.linalg.inv(A) 
x_inv = A_inv.dot(b)



print("\nx_solve (rozkład LU) - wyniki zbliżone do 1:")
print(np.array2string(x_solve, formatter={'float_kind': lambda x: f"{x: .6e}"}))

print("\nx_lstsq (metoda najmniejszych kwadratów) - troche wieksze odchylenia:")
print(np.array2string(x_lstsq, formatter={'float_kind': lambda x: f"{x: .6e}"}))

print("\nx_inv (z użyciem A^{-1} – bardzo duże odchylenia od poprawnego wyniku):")
print(np.array2string(x_inv, formatter={'float_kind': lambda x: f"{x: .6e}"}))



# Wskaźnik uwarunkowania - Określa jak wrażliwe jest rozwiązanie układu równań Ax = b na błędy numeryczne.
condA = np.linalg.cond(A)
print("\ncond(A) =", condA)
# print("wskaźnik uwarunkowania bardzo wyoski dla Hilberta, 10¹³ tracimy około 13 cyfr dokładności,")


# Różnice względem solve
err_lstsq = np.linalg.norm(x_lstsq - x_solve)
err_inv = np.linalg.norm(x_inv - x_solve)

print("\nRóżnice względem solve:")
print("||x_lstsq - x_solve|| =", err_lstsq)
print("||x_inv   - x_solve|| =", err_inv)


"""
1. Co oznacza norma ||x_lstsq – x_solve|| ≈ 3.17 × 10⁻⁵ ?
rozwiązanie lstsq() różni się od idealnego rozwiązania tylko o ok. 0.00003, to są odchylenia na poziomie 5. miejsca po przecinku,

 2. Co oznacza norma ||x_inv – x_solve|| ≈ 0.00879 ?
0.0088≈0.9% błędu

rozwiązanie uzyskane przez obliczenie odwrotności (inv(A).dot(b)) odbiega od idealnego o prawie 1%,

to potwierdza, że metoda z odwrotnością jest bardzo niestabilna numerycznie.
"""