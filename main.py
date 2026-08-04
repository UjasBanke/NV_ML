import os

from train import train_model
from evaluate import evaluate_model

from export import (
    export_metrics,
    export_predictions,
    export_trajectory
)

from plots import (
    plot_feature_importance,
    plot_latitude_error,
    plot_longitude_error,
    plot_altitude_error,
    plot_residuals,
    plot_trajectory,
    plot_actual_vs_predicted
)


os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


def main():
    model, scaler, X_test, y_test = train_model()

    predictions, metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    plot_feature_importance(model)
    plot_latitude_error(y_test, predictions)
    plot_longitude_error(y_test, predictions)
    plot_altitude_error(y_test, predictions)
    plot_residuals(y_test, predictions)
    plot_trajectory(y_test, predictions)
    plot_actual_vs_predicted(y_test, predictions)

    export_metrics(metrics)
    export_predictions(predictions)
    export_trajectory(y_test, predictions)

    print("\nTraining completed successfully.\n")
    print(metrics)


if __name__ == "__main__":
    main()