import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generowanie danych
np.random.seed(42)

n = 100
temp = np.random.randint(0, 31, size=n)
epsilon = np.random.randint(-10, 11, size=n)
sales = 5.0 * temp + 50 + epsilon

data = pd.DataFrame({
    "temp": temp,
    "sales": sales
})

print("First 10 rows:")
print(data.head(10))

# 2. Podział na zbiór treningowy i testowy
X = data[["temp"]]
y = data["sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Dopasowanie modelu regresji liniowej
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predykcja i metryki
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel parameters:")
print(f"Slope (coefficient): {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
print(f"MSE on test set: {mse:.3f}")
print(f"R2 on test set: {r2:.3f}")

# 5. Wykres
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label="Training data", alpha=0.7)
plt.scatter(X_test, y_test, label="Test data", alpha=0.7)
plt.plot(X_train, model.predict(X_train), color="red", label="Fitted line (train)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice cream sales (scoops/day)")
plt.title("Ice Cream Sales vs Temperature")
plt.legend()
plt.tight_layout()
plt.show()

# 6. Interpretacja
print("\nInterpretation:")
print(f"Each additional 1°C increases expected sales by about {model.coef_[0]:.2f} scoops.")
print(f"When temperature is 0°C, expected sales are about {model.intercept_:.2f} scoops.")
print("High R2 indicates that temperature explains most of the variability in sales.")
