import numpy as np
import scipy.linalg as la
import time

#Macierz tridiagonalna symetryczna 
n = 500 
d = 2 * np.ones(n)
a = -1 * np.ones(n-1)

A = np.diag(d) + np.diag(a, 1) + np.diag(a, -1)

# eig() – metoda ogólna
t0 = time.perf_counter()
eigvals_general, _ = la.eig(A) 
t1 = time.perf_counter()


#zapisujemy tylko rzeczywiste części (możliwe fluktuacje w części urojonej ~1e-15)
eigvals_general = np.real(eigvals_general)

time_general = t1 - t0 

# Wartości własne metodą specjalną dla symetrycznych (eigvalsh)
t0 = time.perf_counter()
eigvals_sym = la.eigvalsh(A)
t1 = time.perf_counter()

time_sym = t1 - t0

# difference to miara różnicy między dwoma zestawami wartości własnych
"""One powinny być takie same, ale:
eig() może dawać minimalne błędy numeryczne (bo nie wykorzystuje symetrii)
eigvalsh() jest dużo dokładniejsze dla takich macierzy"""
difference = np.linalg.norm(np.sort(eigvals_general) - np.sort(eigvals_sym))

print("=== CZASY WYKONANIA ===")
print(f"scipy.linalg.eig      : {time_general:.6f} s")
print(f"scipy.linalg.eigvalsh : {time_sym:.6f} s")

print("\n=== RÓŻNICE W WYNIKACH ===")
print(f"Norma różnicy wartości własnych: {difference:.3e}")

print("\n=== KILKA WARTOŚCI WŁASNYCH (metoda ogólna) ===")
print(eigvals_general[:10])  

print("\n=== KILKA WARTOŚCI WŁASNYCH (metoda symetryczna) ===")
print(eigvals_sym[:10])

