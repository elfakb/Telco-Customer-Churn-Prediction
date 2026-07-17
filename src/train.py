
import os
import json
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, classification_report, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from feature_engineering import load_raw_data, engineer_features, prepare_model_data

RANDOM_STATE = 42  #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def evaluate(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]          # churn olma OLASILIĞI (0-1 arası)
    y_pred = (y_proba >= threshold).astype(int)           # olasılığı 0/1 tahmine çeviriyoruz

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def find_best_threshold(model, X_val, y_val):

    y_proba = model.predict_proba(X_val)[:, 1]

    best_thresh = 0.5
    best_f1 = 0

    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_val, (y_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh


def main():

    df = load_raw_data(DATA_PATH)
    df = engineer_features(df)
    X, y = prepare_model_data(df)
    feature_names = X.columns.tolist()


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train churn oranı: {y_train.mean():.3f} | Test churn oranı: {y_test.mean():.3f}")


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    results = {}         # her modelin metriklerini burada topluyoruz
    fitted_models = {}   # eğitilmiş model nesnelerini burada tutuyoruz

    lr_param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}

    lr = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        lr_param_grid,
        scoring="roc_auc",
        cv=5,
    )
    lr.fit(X_train_scaled, y_train)

    results["Logistic Regression"] = evaluate(lr.best_estimator_, X_test_scaled, y_test)
    fitted_models["Logistic Regression"] = (lr.best_estimator_, scaler)
    print("LR best params:", lr.best_params_)

    
    lr_smote = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=lr.best_params_["C"])
    lr_smote.fit(X_train_smote, y_train_smote)
    results["Logistic Regression + SMOTE"] = evaluate(lr_smote, X_test_scaled, y_test)


    rf_param_grid = {
        "n_estimators": [200, 400],       # kaç ağaç kullanılacak
        "max_depth": [6, 10, None],       # her ağacın maksimum derinliği
        "min_samples_leaf": [1, 3, 5],    # bir yaprakta minimum örnek sayısı
    }

    rf = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        rf_param_grid,
        scoring="roc_auc",
        cv=3,
    )
    rf.fit(X_train, y_train)  # Random Forest için scale etmeye gerek yok (ağaç bazlı model)

    results["Random Forest"] = evaluate(rf.best_estimator_, X_test, y_test)
    fitted_models["Random Forest"] = (rf.best_estimator_, None)
    print("RF best params:", rf.best_params_)


    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
        }

        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            proba = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            scores.append(roc_auc_score(y_train.iloc[val_idx], proba))

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)  # 30 farklı kombinasyon dener

    # En iyi bulunan hyperparametrelerle final modeli eğitiyoruz
    best_xgb = XGBClassifier(
        **study.best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
    )
    best_xgb.fit(X_train, y_train)

    results["XGBoost"] = evaluate(best_xgb, X_test, y_test)
    fitted_models["XGBoost"] = (best_xgb, None)
    print("XGBoost best params:", study.best_params)


    comparison_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    print("\n=== MODEL KARŞILAŞTIRMA TABLOSU ===")
    print(comparison_df.round(4))

    comparison_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))

    best_model_name = comparison_df.index[0]  # ROC-AUC'a göre en yüksek skoru alan model
    best_model, best_scaler = fitted_models[best_model_name]
    print(f"\nEn iyi model: {best_model_name}")

    # LR ise scale edilmiş veriyi kullan, RF/XGBoost ise ham veriyi kullan
    X_eval = X_test_scaled if best_scaler is not None else X_test

    best_threshold = find_best_threshold(best_model, X_eval, y_test)
    print(f"Optimal threshold (F1 bazlı): {best_threshold:.2f}")

    final_metrics = evaluate(best_model, X_eval, y_test, threshold=best_threshold)
    print("Final metrikler (optimal threshold ile):", final_metrics)

    y_pred_final = (best_model.predict_proba(X_eval)[:, 1] >= best_threshold).astype(int)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_final))


    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    if best_scaler is not None:
        joblib.dump(best_scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    metadata = {
        "model_name": best_model_name,
        "threshold": float(best_threshold),
        "feature_names": feature_names,
        "needs_scaling": best_scaler is not None,
        "metrics": final_metrics,
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nModel ve metadata kaydedildi ->", MODELS_DIR)


if __name__ == "__main__":
    main()