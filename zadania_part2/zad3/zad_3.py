import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("clients.csv")

# Cechy
features = ["age", "annual_spending", "visits_per_month", "avg_basket_value"]
X = data[features].copy()
y_true = data["true_segment"].copy()  # etykieta pom.

# Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # model k-średnich, dopasowanie na danych przeskalowanych
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_scaled)  

labels = kmeans.labels_  # etykiety klastrów
data["cluster"] = labels  # etykiety do tabeli

# środek klastra
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

centroids_df = pd.DataFrame(centroids_original, columns=features)
centroids_df["cluster"] = np.arange(k) 

# Metryki
inertia = kmeans.inertia_  # rozrzucenie klientów w klastrze, im mniejsze, tym bardzie zwarty klaster
sil = silhouette_score(X_scaled, labels) # 1 super podzial, 0 grupy sie mieszaja, czy klient jest blizej swojej grupy czy innych

# % zgodnych przypisań vs true_segment, zgodnosć rzeczywistego podziału z podziałem modelu
mapping = {}
for cl in range(k):
    mask = labels == cl
    if mask.sum() > 0:
        mapping[cl] = int(y_true[mask].mode()[0])  # najczęstsza etykieta true_segment w danym klastrze

mapped_labels = np.vectorize(mapping.get)(labels)
agreement_pct = (mapped_labels == y_true.values).mean() * 100.0

#  WSS - jak bardzo klient rozni sie od typowego klienta w klastrze (jakość modelu bez standaryzacji) 
# suma kwadratów odległości między klienem a jego centroidem w oryginalnej skali
X_original = X.values
assigned_centroids_original = centroids_original[labels]  # typowu klient
sq_dists_original = np.sum((X_original - assigned_centroids_original) ** 2, axis=1) #kwadrat odlełości, jak bardzo konkretny klient różni się od swojego centroidu
wss_original = float(np.sum(sq_dists_original))  # suma dla wszystkich klientow

# k-NN - wykorzystanie segmentacji dka nowego klientów
knn = KNeighborsClassifier(n_neighbors=5) #patrz na 5 najbardziej podobnych klientów i na tej podstawie zgadnij klaster nowego klienta.
knn.fit(X_scaled, labels) # Uczymy ten model:wejście: dane klientów po skalowaniu,wyjście: numer klastra z k-means.


new_customer = {"age": 33, "annual_spending": 2900, "visits_per_month": 5, "avg_basket_value": 360}
new_customer_df = pd.DataFrame([new_customer], columns=features)
new_customer_scaled = scaler.transform(new_customer_df)  # standaryzacja nowego klienta

pred_cluster = int(knn.predict(new_customer_scaled)[0]) 

# Odległość nowego klienta do centroidu (w skali standaryzowanej i oryginalnej)
centroid_scaled_pred = centroids_scaled[pred_cluster] 
dist_scaled = float(np.linalg.norm(new_customer_scaled[0] - centroid_scaled_pred))

centroid_original_pred = centroids_original[pred_cluster]  
dist_original = float(np.linalg.norm(new_customer_df.values[0] - centroid_original_pred))

# ===== Wyniki =====
print("\n=== KMEANS: wyniki ===")
print(f"Liczba klastrów (k): {k}")
print(f"Inertia (na danych standaryzowanych): {inertia:.3f}")
print(f"Silhouette score (na danych standaryzowanych): {sil:.3f}")
print(f"% zgodnych przypisań vs true_segment (po mapowaniu etykiet): {agreement_pct:.2f}%")
print(f"WSS w oryginalnych jednostkach: {wss_original:.3f}")

print("\nMapowanie klaster -> true_segment (na podstawie majority vote):")
print(mapping)

print("\nCentra klastrów w oryginalnych jednostkach:")
print(centroids_df[["cluster"] + features].to_string(index=False))

print("\n=== k-NN: nowy klient ===")
print("Nowy klient (oryginalna skala):", new_customer)
print(f"Przewidywany klaster: {pred_cluster}")

print("\nCentroid przypisanego klastra (oryginalna skala):")
print(pd.Series(centroid_original_pred, index=features).to_string())

print(f"\nOdległość nowego klienta do centroidu (skala standaryzowana): {dist_scaled:.3f}")
print(f"Odległość nowego klienta do centroidu (skala oryginalna): {dist_original:.3f}")
