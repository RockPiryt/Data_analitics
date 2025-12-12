import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# --- 1. Definicja wielomianu p(x) ---
def p(x):
    return x**4 - 3*x**3 + 2*x - 1

# --- 2. Rysowanie wykresu ---
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
plt.show()

# --- 3. Szukanie przedziałów zmiany znaku ---
xs = np.linspace(-5, 5, 2001)
sign_change_intervals = []

for i in range(len(xs) - 1):
    x1, x2 = xs[i], xs[i+1]
    y1, y2 = p(x1), p(x2)
    
    if y1 == 0:
        sign_change_intervals.append((x1, x1))
    elif y1 * y2 < 0:
        sign_change_intervals.append((x1, x2))

# usunięcie bliskich duplikatów
unique_intervals = []
for a, b in sign_change_intervals:
    if not unique_intervals or abs(a - unique_intervals[-1][0]) > 1e-6:
        unique_intervals.append((a, b))

print("Znalezione przedziały zmiany znaku:")
for a, b in unique_intervals:
    print(f"[{a:.4f}, {b:.4f}]")

# --- 4. Zastosowanie metody Brentq ---
roots = []

for a, b in unique_intervals:
    if a == b:  # idealne trafienie w pierwiastek
        roots.append(a)
    else:
        root = brentq(p, a, b)
        roots.append(root)

# --- 5. Wyniki ---
print("\nRzeczywiste pierwiastki wielomianu:")
for r in roots:
    print(f"x ≈ {r:.10f}")
