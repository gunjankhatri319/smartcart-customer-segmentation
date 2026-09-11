import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def run_pipeline():
  # Ensure the output directory exists
  os.makedirs("models", exist_ok=True)

  # 1. Load data
  data_path = os.path.join("data", "smartcart_customers.csv")
  if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Could not find dataset at '{data_path}'. Please check your data"
        " folder."
    )

  df = pd.read_csv(data_path)

  # 2. Imputation & Outlier Handling
  df["Income"] = df["Income"].fillna(df["Income"].median())
  df["Age"] = 2026 - df["Year_Birth"]
  df = df[(df["Age"] < 90) & (df["Income"] < 200000)].copy()

  # 3. Domain Feature Engineering
  df["Total_Spent"] = df[[
      "MntWines",
      "MntFruits",
      "MntMeatProducts",
      "MntFishProducts",
      "MntSweetProducts",
      "MntGoldProds",
  ]].sum(axis=1)

  df["Total_Purchases"] = (
      df["NumWebPurchases"]
      + df["NumCatalogPurchases"]
      + df["NumStorePurchases"]
  )

  df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
  df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
  df["Customer_Days"] = (
      pd.to_datetime("2026-01-01") - df["Dt_Customer"]
  ).dt.days

  feature_cols = [
      "Income",
      "Recency",
      "Total_Spent",
      "Total_Purchases",
      "NumDealsPurchases",
      "NumWebVisitsMonth",
      "Total_Children",
      "Customer_Days",
  ]
  X = df[feature_cols]

  # 4. Feature Standardization
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # 5. Dimensionality Reduction (2D Projection for UI Dashboard)
  pca = PCA(n_components=2, random_state=42)
  pca_transformed = pca.fit_transform(X_scaled)

  # 6. KMeans Clustering
  kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
  labels = kmeans.fit_predict(X_scaled)

  df["Cluster"] = labels
  df["PCA1"] = pca_transformed[:, 0]
  df["PCA2"] = pca_transformed[:, 1]

  # 7. Unsupervised Evaluation Metrics
  sil_score = silhouette_score(X_scaled, labels)
  ch_score = calinski_harabasz_score(X_scaled, labels)
  db_score = davies_bouldin_score(X_scaled, labels)

  print(f"Silhouette Score: {sil_score:.3f}")
  print(f"Calinski-Harabasz Score: {ch_score:.2f}")
  print(f"Davies-Bouldin Score: {db_score:.3f}")

  # 8. Export Model Artifacts and Segmented Data
  joblib.dump(scaler, "models/scaler.pkl")
  joblib.dump(pca, "models/pca.pkl")
  joblib.dump(kmeans, "models/kmeans_model.pkl")
  df.to_csv("models/clustered_data.csv", index=False)
  print("Training pipeline finished successfully! Artifacts saved to models/.")


if __name__ == "__main__":
  run_pipeline()