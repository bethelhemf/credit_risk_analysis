import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, y_prob):
    """Calculates all required metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def train_and_track():
    # 1. Load Data
    df = pd.read_csv('data/processed/model_ready.csv')
    
    # Select features (Ensure these were engineered in Task 3/4)
    features = ['Total_Value', 'Avg_Value', 'Transaction_Count', 'Std_Value', 'Recency']
    X = df[features]
    y = df['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Define Models and Hyperparameters
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {
            'C': [0.1, 1.0, 10.0]
        }),
        "RandomForest": (RandomForestClassifier(), {
            'n_estimators': [100, 200],
            'max_depth': [5, 10]
        })
    }
    
    mlflow.set_experiment("Bati_Bank_Credit_Risk")

    for model_name, (model, params) in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Training {model_name}...")
            
            # Hyperparameter Tuning
            grid = GridSearchCV(model, params, cv=3, scoring='f1')
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = evaluate_model(y_test, y_pred, y_prob)
            
            # --- MLflow Logging ---
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, model_name)
            
            logger.info(f"{model_name} logged to MLflow with F1: {metrics['f1']:.4f}")

if __name__ == "__main__":
    train_and_track()
    # Identify the best run via MLflow search and register it
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Bati_Bank_Credit_Risk")
runs = client.search_runs(experiment.experiment_id, order_by=["metrics.f1 DESC"], max_results=1)

if runs:
    best_run_id = runs[0].info.run_id
    model_uri = f"runs:/{best_run_id}/RandomForest"
    mlflow.register_model(model_uri, "BatiBank_Credit_Risk_Model")
    logger.info(f"Registered best model from run: {best_run_id}")