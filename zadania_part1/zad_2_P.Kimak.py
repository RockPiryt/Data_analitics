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
# szum
np.random.seed(0)


# oryginalna krzywa y = sin(2πx), x_true ma 400 punktów,
x_true = np.linspace(0, 1, 400)
y_true = np.sin(2*np.pi*x_true)

n = 15 # liczba węzłow
x = np.linspace(0, 1, n) #x
y_clean = np.sin(2*np.pi*x) 

# dodanie szumu
sigma = 0.1
y_noisy = y_clean + np.random.normal(0, sigma, n)# dane z szumem

# Stopnie wielomianów
degrees = [1, 2, 3]
# Słownik 
# coeffs - wspołczynniki wielomianu
# y_fit -  wartosci wwielomianu w pkt pomiarowych
fits = {}

for deg in degrees:
    coeffs = np.polyfit(x, y_noisy, deg)
    y_fit = np.polyval(coeffs, x)
    fits[deg] = dict(coeffs=coeffs, y_fit=y_fit)

# Obliczenie wielkości błędu
def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true)**2))

errors = {deg: rmse(fits[deg]["y_fit"], y_clean) for deg in degrees}

print("Błędy RMSE:")
for deg in degrees:
    print(f"Stopień {deg}: RMSE = {errors[deg]:.4f}")



# Wykres 
plt.figure(figsize=(10, 6))
plt.plot(x_true, y_true, 'k-', label="sin(2πx)") # Oryginalna funkcja
plt.scatter(x, y_noisy, color='red', label="obserwacje z szumem") # Dane z szumem

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
