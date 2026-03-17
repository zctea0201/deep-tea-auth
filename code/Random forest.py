import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Machine Learning & Evaluation
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
import autosklearn.classification
import autosklearn.metrics

# Visualization
from yellowbrick.classifier import (
    ClassificationReport, 
    ConfusionMatrix, 
    ROCAUC, 
    PrecisionRecallCurve
)

# ==========================================
# 1. Configuration & Environment Setup
# ==========================================

# Using Pathlib for cross-platform compatibility and cleaner syntax
WORK_DIR = Path("/home/zctea/A10-Project/2022_TGY/LC_2021/08.machine_learning/")
DATA_FILE = WORK_DIR / "data_original/data_raw.csv"

# Classification Tags: 0: Spring_NS, 1: Spring_QS, 2: Autumn_NS, 3: Autumn_QS
CLASS_TAGS = ["Spring_NS", "Spring_QS", "Autumn_NS", "Autumn_QS"]


# Output directory structure
OUT_DIR = WORK_DIR / f"random_forest/ak_seed2010_t1200"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================

def load_data(file_path, target_col="class"):
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    df = pd.read_csv(file_path, low_memory=False).set_index("label")
    
    # Split training and testing sets based on the "group" column
    train_df = df[df["group"] == "train"]
    test_df = df[df["group"] == "test"]
    
    # Feature extraction (Assuming features start from the 3rd column)
    X_train = train_df.iloc[:, 2:].astype(float)
    y_train = train_df[target_col].astype(int)
    
    X_test = test_df.iloc[:, 2:].astype(float)
    y_test = test_df[target_col].astype(int)
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data(DATA_FILE)

# ==========================================
# 3. Model Training
# ==========================================

automl = autosklearn.classification.AutoSklearnClassifier(
    include={'classifier': ["random_forest"]},
    time_left_for_this_task=1200,
    per_run_time_limit=60,  
    initial_configurations_via_metalearning=0,
    metric=autosklearn.metrics.balanced_accuracy,
    memory_limit=10240,        
    resampling_strategy="cv",
    resampling_strategy_arguments={"folds": 5},
    ensemble_kwargs={'ensemble_size': 1},
    n_jobs=40,
    seed=2010,
    tmp_folder=str(OUT_DIR / "tmp"),
    delete_tmp_folder_after_terminate=False
)

print("Starting model search...")
automl.fit(X_train, y_train)

# Persist the trained model
joblib.dump(automl, OUT_DIR / "best_automl_model.pkl")

# ==========================================
# 4. Prediction & Performance Evaluation
# ==========================================

y_pred = automl.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy Score: {test_acc:.4f}")

# Save detailed evaluation metrics
with open(OUT_DIR / 'prediction_accuracy.txt', 'w') as f:
    f.write(f"Test Accuracy Score: {test_acc}\n\n")
    f.write("Full Classification Report:\n")
    f.write(metrics.classification_report(y_test, y_pred, target_names=CLASS_TAGS))

# ==========================================
# 5. Visualization
# ==========================================


def save_visualizer(visualizer, out_path, X_t, y_t, X_v, y_v):
    """Utility function to fit, score, and save Yellowbrick visualizers"""
    visualizer.fit(X_t, y_t)
    visualizer.score(X_v, y_v)
    visualizer.show(out_path=str(out_path))
    print(f"Visualization saved to: {out_path}")

# Plot 1: Classification Report
save_visualizer(
    ClassificationReport(automl, classes=CLASS_TAGS, support=True, title="Classification Report"),
    OUT_DIR / "Plot_ClassReport.pdf", 
    X_train, y_train, X_test, y_test
)

# Plot 2: Confusion Matrix
save_visualizer(
    ConfusionMatrix(automl, classes=CLASS_TAGS, title="Confusion Matrix"),
    OUT_DIR / "Plot_ConfusionMatrix.pdf",
    X_train, y_train, X_test, y_test
)

# Plot 3: ROC-AUC Curves
save_visualizer(
    ROCAUC(automl, classes=CLASS_TAGS, title="ROCAUC Curves"),
    OUT_DIR / "Plot_ROCAUC.pdf",
    X_train, y_train, X_test, y_test
)

# Plot 4: Precision-Recall Curve
save_visualizer(
    PrecisionRecallCurve(automl, title="Precision-Recall Curves"),
    OUT_DIR / "Plot_PrecisionRecallCurve.pdf",
    X_train, y_train, X_test, y_test
)

print(f"\nAnalysis completed successfully. Results are stored in: {OUT_DIR}")