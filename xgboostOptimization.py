import pandas as pd
import numpy as np
import optuna
import mlflow
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, ConfusionMatrixDisplay, precision_score, recall_score
import pickle
import matplotlib.pyplot as plt
import os

def model_eval(y_test, y_pred):
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc_score", roc_auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y_test))
    disp.plot(cmap='cividis')

    cm_filename = "confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()
    mlflow.log_artifact(cm_filename)
    os.remove(cm_filename)

    return f1


# Load data
with open('proccesedData.pkl', 'rb') as handle:
    proccesedData = pickle.load(handle)

X_train, X_test, y_train, y_test = (
    proccesedData["X_train"],
    proccesedData["X_test"],
    proccesedData["y_train"],
    proccesedData["y_test"],
)

def objective(trial):
    with mlflow.start_run(nested=True, run_name=f"Trial {trial.number}") as child_run:
        param = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True),
        }

        mlflow.log_params(param)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(param, dtrain, num_boost_round=100)

        y_pred = model.predict(X_test)
        
        f1 = model_eval(y_test, y_pred)
        return f1


study_name = "telco_xgboost_optimization"
study = optuna.create_study(direction="maximize")
mlflow.set_experiment(study_name)
with mlflow.start_run(run_name="Parent Run") as parent_run:
    mlflow.set_tag("mlflow.runName", "Parent_Run")
    mlflow.set_tag("optuna_study_name", study_name)

    study.optimize(objective, n_trials=3)

    best_trial = study.best_trial
    mlflow.log_params(best_trial.params)

    best_params = best_trial.params
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    best_model = xgb.train(best_params, dtrain, num_boost_round=100)

    y_pred_prob = best_model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)
    model_eval(y_test, y_pred)