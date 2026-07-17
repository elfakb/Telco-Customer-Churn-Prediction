
import json
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_engineering import load_raw_data, engineer_features, prepare_model_data

REPORTS_DIR = "../reports"
MODELS_DIR = "../models"


def load_artifacts():
    model = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    with open(f"{MODELS_DIR}/metadata.json") as f:
        metadata = json.load(f)
    scaler = None
    if metadata["needs_scaling"]:
        scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    return model, scaler, metadata


def main():
    model, scaler, metadata = load_artifacts()

    df = load_raw_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = engineer_features(df)
    X, y = prepare_model_data(df)
    X = X[metadata["feature_names"]]  

    X_shap = X.copy()
    if scaler is not None:
        X_shap = pd.DataFrame(scaler.transform(X), columns=X.columns)

    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list): 
            shap_values = shap_values[1]
    else:
        explainer = shap.Explainer(model, X_shap)
        shap_values = explainer(X_shap).values

    # 1) Global özet: en etkili feature'lar (
    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {metadata['model_name']}")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_feature_importance.png", dpi=150)
    plt.close()

    # 2) Detaylı dağılım (beeswarm): her feature'ın değeri churn'ü nasıl etkiliyor
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title(f"SHAP Summary Plot - {metadata['model_name']}")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_summary_beeswarm.png", dpi=150)
    plt.close()

    # 3) En etkili 10 feature'ı tablo olarak da kaydet (README'de kullanmak için)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_shap.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).head(10)
    importance_df.to_csv(f"{REPORTS_DIR}/shap_top10_features.csv", index=False)
    print("En etkili 10 feature:\n", importance_df)

    # 4) Örnek bir "yüksek riskli" müşteri için local açıklama (waterfall)
    y_proba = model.predict_proba(X_shap)[:, 1]
    high_risk_idx = np.argmax(y_proba)

    plt.figure()
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        expl = shap.Explanation(
            values=shap_values[high_risk_idx],
            base_values=explainer.expected_value,
            data=X_shap.iloc[high_risk_idx],
            feature_names=X_shap.columns.tolist(),
        )
    else:
        expl = explainer(X_shap)[high_risk_idx]

    shap.plots.waterfall(expl, show=False)
    plt.title(f"Yüksek Riskli Müşteri - Churn Olasılığı: {y_proba[high_risk_idx]:.2%}")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_waterfall_example.png", dpi=150)
    plt.close()

    print("\nSHAP görselleri kaydedildi -> reports/")
    print(" - shap_feature_importance.png")
    print(" - shap_summary_beeswarm.png")
    print(" - shap_waterfall_example.png")
    print(" - shap_top10_features.csv")


if __name__ == "__main__":
    main()