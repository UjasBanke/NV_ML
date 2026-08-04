import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from preprocess import preprocess_data


DATA_PATH = "data/final_nv_navigation_dataset.csv"


def train_model():
    X_train, X_test, y_train, y_test, scaler = preprocess_data(DATA_PATH)

    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/xgboost_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return model, scaler, X_test, y_test