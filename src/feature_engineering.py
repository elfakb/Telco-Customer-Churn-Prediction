import pandas as pd
import numpy as np


def load_raw_data(path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    # total charges sütununda tüm değerlle numeric değil bazıları boş string 
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # boş olanları da 0 ile doldurucaz
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


def engineer_features(df):
    df = df.copy()  
    # tenure group için farklı zaman dilimlerine ayırırız
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6ay", "6-12ay", "1-2yil", "2-4yil", "4-6yil"],
        include_lowest=True,
    )

    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # faturalardaki fiyat değişimi çok fazla ise churn olasılığı artar 
    df["monthly_charge_change_rate"] = np.where(
        df["avg_monthly_spend"] > 0,
        (df["MonthlyCharges"] - df["avg_monthly_spend"]) / df["avg_monthly_spend"],
        0,
    )

    # 4) müşteri ne kadar fazla ek servis almışsa o kadar bağlı olur yani churn düşer
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["total_services"] = (df[service_cols] == "Yes").sum(axis=1)

    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["has_no_tech_support"] = (df["TechSupport"] == "No").astype(int)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    return df


def prepare_model_data(df):
    df = df.drop(columns=["customerID"], errors="ignore")

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X, y


if __name__ == "__main__":
    df = load_raw_data()
    df = engineer_features(df)
    X, y = prepare_model_data(df)

    print("Toplam feature sayısı:", X.shape[1])
    print("Churn oranı:", round(y.mean(), 3))
    print(X.head())