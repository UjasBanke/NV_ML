import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/final_nv_navigation_dataset.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    target_columns = [
        "lat_deg",
        "lon_deg",
        "alt_m"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in target_columns + ["time_s"]
    ]

    X = df[feature_columns]
    y = df[target_columns]
    time = df["time_s"]

    return X, y, time


def preprocess_data():
    X, y, time = load_data()

    X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
        X,
        y,
        time,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_all_scaled = scaler.transform(X)

    return (
        X_train_scaled,
        X_test_scaled,
        X_all_scaled,
        y_train,
        y_test,
        y,
        time_train,
        time_test,
        time
    )