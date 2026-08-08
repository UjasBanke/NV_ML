import json
import pandas as pd

from plots import calculate_3d_position_error


def export_metrics(metrics):

    with open(
        "outputs/metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )


def export_predictions(
    predictions
):

    dataframe = pd.DataFrame(
        predictions,
        columns=[
            "lat_deg",
            "lon_deg",
            "alt_m"
        ]
    )

    dataframe.to_csv(
        "outputs/predictions.csv",
        index=False
    )


def export_trajectory(
    y_test,
    predictions
):

    trajectory = []

    for i in range(
        len(predictions)
    ):

        point = {
            "actual_lat": float(
                y_test.iloc[i, 0]
            ),

            "actual_lon": float(
                y_test.iloc[i, 1]
            ),

            "actual_alt": float(
                y_test.iloc[i, 2]
            ),

            "predicted_lat": float(
                predictions[i, 0]
            ),

            "predicted_lon": float(
                predictions[i, 1]
            ),

            "predicted_alt": float(
                predictions[i, 2]
            )
        }

        trajectory.append(
            point
        )

    with open(
        "outputs/trajectory.json",
        "w"
    ) as file:

        json.dump(
            trajectory,
            file,
            indent=4
        )


def export_3d_position_error(
    y_all,
    predictions_all,
    time_all
):

    (
        lat_error_m,
        lon_error_m,
        alt_error_m,
        position_error_3d_m
    ) = calculate_3d_position_error(
        y_all,
        predictions_all
    )

    error_data = []

    for i in range(
        len(predictions_all)
    ):

        point = {
            "time_s": float(
                time_all.iloc[i]
            ),

            "latitude_error_m": float(
                lat_error_m[i]
            ),

            "longitude_error_m": float(
                lon_error_m[i]
            ),

            "altitude_error_m": float(
                alt_error_m[i]
            ),

            "position_error_3d_m": float(
                position_error_3d_m[i]
            )
        }

        error_data.append(
            point
        )

    error_data.sort(
        key=lambda x: x["time_s"]
    )

    with open(
        "outputs/3d_position_error.json",
        "w"
    ) as file:

        json.dump(
            error_data,
            file,
            indent=4
        )