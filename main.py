import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. Load Dataset
# ==============================
print("Isi folder:", os.listdir())

df = pd.read_csv("online_gaming_behavior.csv.csv")

print("\nData:")
print(df.head())

print("\nKolom:")
print(df.columns)

# ==============================
# 2. Pilih fitur
# ==============================
features = [
    'PlayTimeHours',
    'InGamePurchases',
    'SessionsPerWeek',
    'AvgSessionDurationMinutes',
    'PlayerLevel'
]

X = df[features]

# ==============================
# 3. Normalisasi
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData setelah normalisasi:")
print(X_scaled[:5])

# ==============================
# 4. Elbow Method
# ==============================
inertia = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow
plt.figure()
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# ==============================
# 5. K-Means Final (K=5)
# ==============================
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nHasil clustering:")
print(df[['PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek',
          'AvgSessionDurationMinutes', 'PlayerLevel', 'Cluster']].head())

# ==============================
# 6. Visualisasi
# ==============================
plt.figure()
sns.scatterplot(
    x=df['PlayTimeHours'],
    y=df['InGamePurchases'],
    hue=df['Cluster']
)

plt.title('Clustering Gamer')
plt.xlabel('Play Time Hours')
plt.ylabel('In Game Purchases')
plt.show()
