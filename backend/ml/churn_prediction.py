import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle

class ChurnPredictor:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        self.scaler = StandardScaler()
        self.features = [
            "purchase_frequency",
            "recency_days",
            "total_spending",
            "complaints_count",
            "average_review_rating",
            "browsing_sessions"
        ]

    def fit_and_evaluate(self, df: pd.DataFrame) -> dict:
        """
        Fits XGBoost on features list and prints/returns evaluation metrics.
        df must contain self.features and 'churned' target column.
        """
        X = df[self.features].fillna(0)
        y = df["churned"].fillna(0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Fit model
        self.model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba))
        }
        return metrics

    def predict_churn(self, purchase_frequency: float, recency_days: float, total_spending: float, 
                      complaints_count: int, average_review_rating: float, browsing_sessions: int) -> dict:
        """
        Predicts churn probability, risk level, and suggestions for a single customer.
        """
        x_raw = np.array([[purchase_frequency, recency_days, total_spending, complaints_count, average_review_rating, browsing_sessions]])
        x_scaled = self.scaler.transform(x_raw)
        
        proba = float(self.model.predict_proba(x_scaled)[0, 1])
        
        # Risk level assessment
        if proba < 0.3:
            risk_level = "Low Risk"
            suggestions = [
                "Maintain regular engagement newsletters.",
                "Recommend trending and similar products.",
                "Offer standard loyalty points updates."
            ]
        elif proba < 0.8:
            risk_level = "Medium Risk"
            suggestions = [
                "Offer a small discount coupon (10% off) for their favorite category.",
                "Send personalized recommendations based on browsing history.",
                "Ask for feedback on recent browsing sessions to identify friction."
            ]
        else:
            risk_level = "High Risk"
            suggestions = [
                "Trigger the Churn Retention Engine: Send a high-value coupon (20% off).",
                "Award 500 bonus loyalty points to encourage immediate retention.",
                "Send an automated email notification apologizing for any issues, offering direct customer support."
            ]

        return {
            "churn_probability": proba,
            "risk_level": risk_level,
            "suggestions": suggestions
        }

    def save(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> 'ChurnPredictor':
        with open(file_path, "rb") as f:
            return pickle.load(f)
