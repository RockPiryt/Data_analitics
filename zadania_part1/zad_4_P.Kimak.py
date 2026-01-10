
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def p(x):
    return x**4 - 3*x**3 + 2*x - 1

# rys wykresu wielomianu
x_plot = np.linspace(-5, 5, 1000)
y_plot = p(x_plot)

plt.figure(figsize=(9, 5))
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.plot(x_plot, y_plot, label="p(x) = x^4 - 3x^3 + 2x - 1")
plt.title("Wykres wielomianu p(x)")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.grid(True)
plt.legend()
plt.ylim(-10, 10)
plt.show()

# Szukanie przedziałów zmiany znaku
xs = np.linspace(-5, 5, 2001) # 2001 punktoów-> 2000 małych przedziałów o dł 0.0005
sign_change_intervals = []

for i in range(len(xs) - 1):
    x1, x2 = xs[i], xs[i+1] 
    y1, y2 = p(x1), p(x2)
    
    if y1 == 0: # trafilismy dokladnie w pierwiastek
        sign_change_intervals.append((x1, x1))
    elif y1 * y2 < 0: # różne znaki na końcach przedziału, wiec w przedziale jest pierwiatek
        sign_change_intervals.append((x1, x2))

# usunięcie bliskich duplikatów
unique_intervals = []
for a, b in sign_change_intervals:
    if not unique_intervals or abs(a - unique_intervals[-1][0]) > 1e-6:
        unique_intervals.append((a, b))

print("Znalezione przedziały zmiany znaku:")
for a, b in unique_intervals:
    print(f"[{a:.4f}, {b:.4f}]")

# Metoda Brentq
roots = []

for a, b in unique_intervals:
    if a == b: 
        roots.append(a)
    else:
        root = brentq(p, a, b) # szukanie miejsca zerowego w konretnym przedziale, p - funkcja, p(x) - konretna liczba
        roots.append(root)

print("\nRzeczywiste pierwiastki wielomianu:")
for r in roots:
    print(f"x ≈ {r:.10f}")