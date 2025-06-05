import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from sklearn.datasets import load_iris

# Set experiment (auto-creates if it doesn't exist)
mlflow.set_experiment("Iris_Classification")

# Load Iris dataset directly from sklearn
iris = load_iris(as_frame=True)
df = iris.frame
df['target'] = iris.target

# Split data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models to evaluate
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Track best model
best_model_name = None
best_model_score = 0.0
best_model_uri = None

# Train and evaluate models
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log params and metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)

        # Save and log model artifact
        model_path = f"models/{model_name}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Log model for model registry
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Track the best model
        if acc > best_model_score:
            best_model_score = acc
            best_model_name = model_name
            best_model_uri = f"runs:/{run.info.run_id}/model"

# Register the best model
if best_model_uri:
    result = mlflow.register_model(
        model_uri=best_model_uri,
        name="Best_Iris_Model"
    )
    print(f"✅ Registered best model: {best_model_name} (accuracy: {best_model_score:.4f})")
