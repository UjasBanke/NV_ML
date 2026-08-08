import os
import joblib

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from preprocess import preprocess_data


def train_model():

    os.makedirs(
        "models",
        exist_ok=True
    )

    (
        X_train,
        X_test,
        X_all,
        y_train,
        y_test,
        y_all,
        time_train,
        time_test,
        time_all
    ) = preprocess_data()

    base_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model = MultiOutputRegressor(
        base_model
    )

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        "models/xgboost_model.pkl"
    )

    return (
        model,
        X_test,
        y_test,
        X_all,
        y_all,
        time_test,
        time_all
    )