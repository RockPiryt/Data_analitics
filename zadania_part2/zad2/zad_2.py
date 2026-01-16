import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Generowanie danych
np.random.seed(42)
n = 300

area = np.random.uniform(40, 200, size=n)
people = np.random.choice([1, 2, 3, 4, 5], size=n)
windows = np.random.randint(1, 11, size=n)
age = np.random.uniform(0, 80, size=n)
epsilon = np.random.normal(0, 20, size=n)

consumption = 0.8 * area + 10 * people + 2 * windows - 0.1 * age + epsilon

# stworzenie tabeli danych
data = pd.DataFrame({
    "area": area,
    "people": people,
    "windows": windows,
    "age": age,
    "consumption": consumption
})

data.to_csv("energy.csv", index=False)
print("Saved data to energy.csv")
print("\nFirst 10 rows:")
print(data.head(10))

# Podział na zbiór treningowy i testowy
X = data[["area", "people", "windows", "age"]]
y = data["consumption"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Dopasowanie modelu bez standaryzacji
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model bez standaryzacji ---")
print(f"MSE: {mse:.3f}")
print(f"R2: {r2:.3f}")

for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

# Standaryzacja cech
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

model_std = LinearRegression()
model_std.fit(X_train_std, y_train)

y_pred_std = model_std.predict(X_test_std)

mse_std = mean_squared_error(y_test, y_pred_std)
r2_std = r2_score(y_test, y_pred_std)

print("\n--- Model ze standaryzacja cech ---")
print(f"MSE: {mse_std:.3f}")
print(f"R2: {r2_std:.3f}")

for name, coef in zip(X.columns, model_std.coef_):
    print(f"{name} (std): {coef:.3f}")
print(f"Intercept (std): {model_std.intercept_:.3f}")

# Zapis wyników do pliku TXT
with open("results.txt", "w", encoding="utf-8") as f:
    f.write("--- Model bez standaryzacji ---\n")
    f.write(f"MSE: {mse:.3f}\n")
    f.write(f"R2: {r2:.3f}\n")

    for name, coef in zip(X.columns, model.coef_):
        f.write(f"{name}: {coef:.3f}\n")
    f.write(f"Intercept: {model.intercept_:.3f}\n\n")

    f.write("--- Model ze standaryzacja cech ---\n")
    f.write(f"MSE: {mse_std:.3f}\n")
    f.write(f"R2: {r2_std:.3f}\n")

    for name, coef in zip(X.columns, model_std.coef_):
        f.write(f"{name} (std): {coef:.3f}\n")
    f.write(f"Intercept (std): {model_std.intercept_:.3f}\n")

print("\nWyniki zapisano do pliku results.txt")

# Interpretaacja:
# - Standaryzacja cech nie zmieniła jakości modelu – 
# wartości MSE i R² są identyczne, 
# ponieważ przeskalowanie zmiennych wejściowych nie wpływa na zdolność predykcyjną regresji liniowej.

# - W modelu bez standaryzacji współczynniki można interpretować w jednostkach rzeczywistych, 
# np. wzrost powierzchni o 1 m² zwiększa zużycie energii o około 0.83 jednostki, 
# a każda dodatkowa osoba o około 9.3 jednostki.

# - Model ze standaryzacją pozwala natomiast porównać znaczenie zmiennych.
# Największy wpływ na zużycie ma powierzchnia mieszkania, następnie liczba osób i liczba okien, a najmniejszy – wiek budynku.
