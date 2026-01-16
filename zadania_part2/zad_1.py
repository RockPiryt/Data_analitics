import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generowanie danych
np.random.seed(42)
n = 100
temp = np.random.randint(0, 31, size=n) #  Temperaturę też losujemy jako liczbę całkowitą z zakresu od 0 do 30.
epsilon = np.random.randint(-10, 11, size=n) # Liczbę ε losujemy jako liczbę całkowitą z zakresu od −10 do 10
sales = 5.0 * temp + 50 + epsilon

# Stworznie tabelę danych (temp /sales) w pandas ze 100 wierszami
data = pd.DataFrame({
    "temp": temp,
    "sales": sales
})

print("First 10 rows:")
print(data.head(10))

# Zapis wygenerowanych danych do pliku CSV
data.to_csv("ice_cream_sales.csv", index=False)

# Podział na zbiór treningowy i testowy
X = data[["temp"]]
y = data["sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Dopasowanie modelu regresji liniowej
model = LinearRegression()
model.fit(X_train, y_train)

# Predykcja i metryki
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred) # metryka błędu przewidywania modelu
r2 = r2_score(y_test, y_pred) 

print("\nModel parameters:")
print(f"Slope (coefficient): {model.coef_[0]:.3f}") # współczynnik kierunkowy
print(f"Intercept: {model.intercept_:.3f}") # wyraz wolny
print(f"MSE on test set: {mse:.3f}") # metryka błędu przwidywania
print(f"R2 on test set: {r2:.3f}")

# Wykres
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label="Training data", alpha=0.7)
plt.scatter(X_test, y_test, label="Test data", alpha=0.7)
plt.plot(X_train, model.predict(X_train), color="red", label="Fitted line (train)")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Sprzedaż lodów(scoops/day)")
plt.title("Przewidywanie sprzedaży na podstawie temperatury")
plt.legend()
plt.tight_layout()
plt.show()

#  Interpretacja
print("\nInterpretacja:")
print(f"Każdy dodatkowy 1°C temperatury zwiększa przewidywaną sprzedaż o około {model.coef_[0]:.2f} gałki.")
print(f"Przy temperaturze 0°C przewidywana sprzedaż wynosi około {model.intercept_:.2f} gałki.")
print("Wysokie R² wskazuje, że zależność między temperaturą a sprzedażą jest silna, a dopasowana prosta dobrze opisuje dane.")
