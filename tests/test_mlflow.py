"""
Script de test MLflow en isolation (sans AWS, sans DataManager).
Simule ce que fait ExpandingWindowScheme après le training.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from pathlib import Path

# Forcer le même tracking URI que l'API (racine du projet)
_ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(str(_ROOT / "mlruns"))

# --- Données fictives ---
np.random.seed(42)
n = 100
X = pd.DataFrame({"feature_1": np.random.randn(n), "feature_2": np.random.randn(n)})
y = pd.DataFrame({"CPIAUCSL": 0.5 * X["feature_1"] - 0.3 * X["feature_2"] + np.random.randn(n) * 0.1})

# --- Simulation du résultat du grid search ---
fake_results = {
    "ols": {
        "model": LinearRegression(),
        "hyperparams": {},
        "val_rmse": 0.12,
    },
    "lasso": {
        "model": Lasso(alpha=0.01),
        "hyperparams": {"alpha": 0.01},
        "val_rmse": 0.11,
    },
}

# Entraîner les modèles sur les données fictives
for name, info in fake_results.items():
    info["model"].fit(X, y.values.ravel())
    y_hat = info["model"].predict(X)
    info["oos_rmse"] = float(np.sqrt(mean_squared_error(y, y_hat)))
    correct_sign = np.sign(y.values.ravel()) == np.sign(y_hat)
    info["sign_accuracy"] = float(correct_sign.mean())

# --- MLflow tracking (même logique que expanding.py) ---
mlflow.set_experiment("forecasting_expanding")
print("\n✅ Expérience MLflow créée : 'forecasting_expanding'")

for model_name, info in fake_results.items():
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("estimation_method", "expanding")
        mlflow.log_param("forecast_horizon", 1)
        mlflow.log_param("validation_window", 12)
        mlflow.log_params(info["hyperparams"])

        mlflow.log_metric("val_rmse", info["val_rmse"])
        mlflow.log_metric("oos_rmse", info["oos_rmse"])
        mlflow.log_metric("sign_accuracy", info["sign_accuracy"])

        mlflow.sklearn.log_model(info["model"], name="model")

        print(f"   → {model_name}: OOS RMSE={info['oos_rmse']:.4f}, Sign accuracy={info['sign_accuracy']:.2%}")

best_model_name = min(fake_results, key=lambda m: fake_results[m]["oos_rmse"])
print(f"\n   Meilleur modèle : {best_model_name}")

best = fake_results[best_model_name]
with mlflow.start_run(run_name=f"registry_{best_model_name}"):
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("oos_rmse", best["oos_rmse"])
    mlflow.log_metric("sign_accuracy", best["sign_accuracy"])
    mlflow.sklearn.log_model(best["model"], name="model", registered_model_name="forecasting_best_model")

client = mlflow.MlflowClient()
latest = client.get_latest_versions("forecasting_best_model")[0]
client.set_registered_model_alias("forecasting_best_model", "production", latest.version)
print(f"   → '{best_model_name}' enregistré en production (version {latest.version})")

print("\n✅ MLflow tracking + registry OK !")
print("   uv run mlflow ui  →  http://localhost:5000")
print("   uv run uvicorn ml_and_backtester_app.dashboard.app:api --port 8050  →  http://localhost:8050/docs")
