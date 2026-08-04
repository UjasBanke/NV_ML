import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "B_proj_D1_nT",
    "B_proj_D2_nT",
    "B_proj_D3_nT",
    "B_proj_D4_nT",
    "Bx_meas_nT",
    "By_meas_nT",
    "Bz_meas_nT",
    "B_total_meas_nT"
]

TARGETS = [
    "lat_deg",
    "lon_deg",
    "alt_m"
]


def load_data(path):
    df = pd.read_csv(path)

    X = df[FEATURES]
    y = df[TARGETS]

    return X, y


def preprocess_data(path):
    X, y = load_data(path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler