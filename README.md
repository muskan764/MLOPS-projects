# 🌸 Iris Classification with MLflow

This project demonstrates **end-to-end machine learning workflow** on the classic **Iris dataset**, using  
- **MLflow** for experiment tracking & model registry  
- **scikit-learn** for training classification models  
- **pandas** for data handling  

The workflow covers dataset preparation, model training, experiment logging, and best model registration.

----------------------------------------------------------------------------

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/iris-mlflow.git
   cd iris-mlflow
Set up a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt
🚀 Usage
1️⃣ Save the Dataset
Generate and save the Iris dataset:
python save_dataset.py
👉 This creates data/iris_dataset.csv.

2️⃣ Train Models & Track with MLflow
Train and log models:
python train.py
✔ Trains RandomForest & GradientBoosting classifiers
✔ Logs metrics, parameters & artifacts to MLflow
✔ Saves models in the models/ folder
✔ Registers the best model automatically

📊 MLflow Tracking UI
To visualize experiments:
mlflow ui
Then open in your browser: http://127.0.0.1:5000

You’ll see:

Parameters (e.g., model type)

Metrics (accuracy)

Artifacts (saved .pkl models)

Registered best model

🛠️ Requirements
Python 3.9+

Dependencies (in requirements.txt):

mlflow==2.12.1

pandas==2.2.2

scikit-learn==1.4.2

joblib==1.4.2

✨ Features
✅ Automatic dataset saving (save_dataset.py)

✅ Training with multiple ML models

✅ Experiment tracking with MLflow

✅ Model artifacts stored locally (models/)

✅ Best model auto-registered in MLflow Model Registry

📌 Next Steps
🔧 Add hyperparameter tuning (GridSearchCV / Optuna)

📈 Try more algorithms (e.g., XGBoost, SVM)

🚀 Deploy registered model with MLflow Serving or FastAPI
