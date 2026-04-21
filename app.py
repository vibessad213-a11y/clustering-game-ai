from flask import Flask, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)


@app.route('/')
def index():
    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("online_gaming_behavior.csv.csv")

    # =========================
    # PILIH FITUR
    # =========================
    features = [
        'PlayTimeHours',
        'InGamePurchases',
        'SessionsPerWeek',
        'AvgSessionDurationMinutes',
        'PlayerLevel'
    ]

    X = df[features]

    # =========================
    # NORMALISASI
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # K-MEANS
    # =========================
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # =========================
    # LABEL CLUSTER (BIAR KEREN)
    # =========================
    cluster_names = {
        0: "Casual Gamer",
        1: "Hardcore Gamer",
        2: "Big Spender",
        3: "Active Player",
        4: "Low Activity"
    }

    df['ClusterName'] = df['Cluster'].map(cluster_names)

    # =========================
    # STATISTIK
    # =========================
    total = len(df)
    avg_hours = round(df['PlayTimeHours'].mean(), 2)
    paying = len(df[df['InGamePurchases'] > 0])

    # =========================
    # TABEL
    # =========================
    table = df.head(20)[[
        'PlayerID',
        'PlayTimeHours',
        'InGamePurchases',
        'SessionsPerWeek',
        'PlayerLevel',
        'Cluster',
        'ClusterName'
    ]].to_html(classes='table table-dark table-hover', index=False)

    return render_template(
        'index.html',
        tables=table,
        total=total,
        avg_hours=avg_hours,
        paying=paying
    )


if __name__ == '__main__':
    app.run(debug=True)
