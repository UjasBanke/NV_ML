import os

from train import train_model

from evaluate import evaluate_model

from export import (
    export_metrics,
    export_predictions,
    export_trajectory,
    export_3d_position_error
)

from plots import (
    plot_feature_importance,
    plot_latitude_error,
    plot_longitude_error,
    plot_altitude_error,
    plot_residuals,
    plot_trajectory,
    plot_actual_vs_predicted,
    plot_3d_position_error
)


os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "outputs",
    exist_ok=True
)


def main():

    (
        model,
        X_test,
        y_test,
        X_all,
        y_all,
        time_test,
        time_all
    ) = train_model()

    predictions_test, metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    predictions_all = model.predict(
        X_all
    )

    plot_feature_importance(
        model
    )

    plot_latitude_error(
        y_test,
        predictions_test
    )

    plot_longitude_error(
        y_test,
        predictions_test
    )

    plot_altitude_error(
        y_test,
        predictions_test
    )

    plot_residuals(
        y_test,
        predictions_test
    )

    plot_trajectory(
        y_test,
        predictions_test
    )

    plot_actual_vs_predicted(
        y_test,
        predictions_test
    )

    plot_3d_position_error(
        y_all,
        predictions_all,
        time_all
    )

    export_metrics(
        metrics
    )

    export_predictions(
        predictions_test
    )

    export_trajectory(
        y_test,
        predictions_test
    )

    export_3d_position_error(
        y_all,
        predictions_all,
        time_all
    )

    print(
        "\nTraining completed successfully.\n"
    )

    print(
        "Test metrics:"
    )

    print(
        metrics
    )


if __name__ == "__main__":
    main()