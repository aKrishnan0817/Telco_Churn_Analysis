import pandas as pd
import numpy as np
import optuna 
import mlflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, ConfusionMatrixDisplay, r2_score, precision_score, recall_score
import pickle
import mlflow.tensorflow
import os
from xgboost import XGBClassifier

def model_eval( y_test, y_pred):
    # Log metrics
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc_score", roc_auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))

    # Generate and log confusion matrix as an artifact
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y_test))
    disp.plot(cmap='cividis')

    # Save confusion matrix plot
    cm_filename = "confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()
    mlflow.log_artifact(cm_filename)
    os.remove(cm_filename)

    return f1


with open('proccesedData.pkl', 'rb') as handle:
    proccesedData = pickle.load(handle)

X_train,X_test,y_train,y_test=proccesedData["X_train"],proccesedData["X_test"],proccesedData["y_train"],proccesedData["y_test"]

counter = 1
def objective(trial):
    global counter
    with mlflow.start_run(nested=True,run_name=f"Trial {trial.number}") as child_run:
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        layer_sizes = [trial.suggest_int(f"layer_size_{i+1}", 16, 128, log=True) for i in range(num_layers)]

        mlflow.log_params({
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            **{f"layer_size_{i+1}": size for i, size in enumerate(layer_sizes)}
        })

        model = Sequential()
        model.add(Dense(layer_sizes[0], activation="relu", input_shape=(X_train.shape[1],)))
        for i in range(1, num_layers):
            model.add(Dense(layer_sizes[i], activation="relu"))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation="sigmoid"))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, verbose=0)

        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        f1_score= model_eval(y_test, y_pred)

        return f1_score

# Start Optuna study and parent MLflow run
study_name = "telco_neuralnetwork_optimization"
study = optuna.create_study(direction="maximize")
mlflow.set_experiment(study_name)
with mlflow.start_run(run_name="Parent Run") as parent_run:
    mlflow.set_tag("mlflow.runName", "Parent_Run")
    mlflow.set_tag("optuna_study_name", study_name)

    study.optimize(objective, n_trials=3)

    # Log best trial
    best_trial = study.best_trial
    mlflow.log_params(best_trial.params)
   # Re-run the best trial to log additional metrics/artifacts
    best_model = Sequential()
    layer_sizes = [best_trial.params[f"layer_size_{i+1}"] for i in range(best_trial.params["num_layers"])]
    dropout_rate = best_trial.params["dropout_rate"]

    best_model.add(Dense(layer_sizes[0], activation="relu", input_shape=(X_train.shape[1],)))
    for size in layer_sizes[1:]:
        best_model.add(Dense(size, activation="relu"))
        best_model.add(Dropout(dropout_rate))
    best_model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=best_trial.params["learning_rate"])
    best_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    best_model.fit(X_train, y_train, epochs=100, batch_size=best_trial.params["batch_size"], validation_split=0.2, verbose=0)

    y_pred = (best_model.predict(X_test) > 0.5).astype("int32")
    model_eval( y_test, y_pred)

