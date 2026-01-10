import numpy as np
import matplotlib.pyplot as plt
"""
2. Dopasowanie wielomianów metodą najmniejszych kwadratów.
• Wygeneruj dane bez szumu: y = sin(2πx) na odcinku [0, 1].
• Wybierz 15 równoodległych węzłów w [0, 1] i wygeneruj obserwacje z szumem (np.
dodaj np.random.normal(0, sigma, size); wybierz rozsądne sigma).
• Dopasuj wielomiany stopnia 1, 2 i 3 metodą najmniejszych kwadratów (np. np.polyfit
lub tworząc macierz Vandermonde i wywołując np.linalg.lstsq).
• Policz błąd dopasowania (np. RMSE na węzłach) i wstaw wykres: oryginalna krzywa
y = sin(2πx), dane z szumem oraz dopasowane wielomiany"""
# --- 1. Dane ---
np.random.seed(0)

# ciągła funkcja referencyjna do rysowania
x_true = np.linspace(0, 1, 400)
y_true = np.sin(2*np.pi*x_true)

# 15 węzłów
n = 15
x = np.linspace(0, 1, n)
y_clean = np.sin(2*np.pi*x)

# dodanie szumu
sigma = 0.1
y_noisy = y_clean + np.random.normal(0, sigma, n)

# --- 2. Dopasowanie wielomianów ---
degrees = [1, 2, 3]
fits = {}

for deg in degrees:
    coeffs = np.polyfit(x, y_noisy, deg)
    y_fit = np.polyval(coeffs, x)
    fits[deg] = dict(coeffs=coeffs, y_fit=y_fit)

# --- 3. Obliczenie RMSE ---
def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true)**2))

errors = {deg: rmse(fits[deg]["y_fit"], y_clean) for deg in degrees}

print("Błędy RMSE:")
for deg in degrees:
    print(f"Stopień {deg}: RMSE = {errors[deg]:.4f}")

# --- 4. Wykres ---
plt.figure(figsize=(10, 6))

# Oryginalna funkcja
plt.plot(x_true, y_true, 'k-', label="sin(2πx)")

# Dane z szumem
plt.scatter(x, y_noisy, color='red', label="obserwacje z szumem")

# Dopasowane wielomiany
for deg in degrees:
    coeffs = fits[deg]["coeffs"]
    y_plot = np.polyval(coeffs, x_true)
    plt.plot(x_true, y_plot, label=f"wielomian stopnia {deg}")

plt.legend()
plt.title("Dopasowanie wielomianów metodą najmniejszych kwadratów")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
