import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class CustomerSegmenter:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        # Map cluster index to business label
        self.cluster_labels = {}

    def fit(self, df_features: pd.DataFrame):
        """
        Fits K-Means on features: ['recency', 'frequency', 'monetary_value', 'tenure']
        df_features should contain customer aggregate data.
        """
        # Features must be numerical
        X = df_features[['recency', 'frequency', 'monetary_value', 'tenure']].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        
        # Determine labels based on cluster center characteristics
        centers = self.kmeans.cluster_centers_
        # Inverse transform to understand actual values
        actual_centers = self.scaler.inverse_transform(centers)
        
        # We assign labels based on rules applied to the cluster centers:
        # centers structure: [recency, frequency, monetary_value, tenure]
        # Let's map each index:
        cluster_info = []
        for idx, center in enumerate(actual_centers):
            cluster_info.append({
                "idx": idx,
                "recency": center[0],
                "frequency": center[1],
                "monetary": center[2],
                "tenure": center[3]
            })
            
        # 1. Premium: High frequency, High monetary, Low recency (active)
        # Sort by monetary descending
        sorted_by_monetary = sorted(cluster_info, key=lambda x: x["monetary"], reverse=True)
        premium_idx = sorted_by_monetary[0]["idx"]
        
        # 2. High-Risk: High recency (haven't bought in a long time), low frequency
        sorted_by_recency = sorted(cluster_info, key=lambda x: x["recency"], reverse=True)
        # Avoid double-tagging premium if premium has high recency (unlikely)
        high_risk_candidates = [c for c in sorted_by_recency if c["idx"] != premium_idx]
        high_risk_idx = high_risk_candidates[0]["idx"] if high_risk_candidates else sorted_by_recency[0]["idx"]
        
        # 3. New Customers: Low tenure, low/medium frequency
        sorted_by_tenure = sorted(cluster_info, key=lambda x: x["tenure"])
        new_candidates = [c for c in sorted_by_tenure if c["idx"] not in (premium_idx, high_risk_idx)]
        new_idx = new_candidates[0]["idx"] if new_candidates else sorted_by_tenure[0]["idx"]
        
        # 4. Regular Customers: Remaining
        used_indices = {premium_idx, high_risk_idx, new_idx}
        regular_candidates = [c for c in cluster_info if c["idx"] not in used_indices]
        regular_idx = regular_candidates[0]["idx"] if regular_candidates else (set(range(self.n_clusters)) - {premium_idx, high_risk_idx, new_idx}).pop()
        
        self.cluster_labels = {
            premium_idx: "Premium",
            high_risk_idx: "High-Risk",
            new_idx: "New Customer",
            regular_idx: "Regular Customer"
        }
        
        return self

    def predict(self, df_features: pd.DataFrame) -> list[str]:
        """Predict business segments for a dataframe of features."""
        X = df_features[['recency', 'frequency', 'monetary_value', 'tenure']].fillna(0)
        X_scaled = self.scaler.transform(X)
        cluster_predictions = self.kmeans.predict(X_scaled)
        
        # Map cluster indices to names
        return [self.cluster_labels.get(c, "Regular Customer") for c in cluster_predictions]
        
    def predict_single(self, recency: float, frequency: float, monetary_value: float, tenure: float) -> str:
        """Predict segment for a single customer sample."""
        X = np.array([[recency, frequency, monetary_value, tenure]])
        X_scaled = self.scaler.transform(X)
        cluster = self.kmeans.predict(X_scaled)[0]
        return self.cluster_labels.get(cluster, "Regular Customer")
