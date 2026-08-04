import json
import pandas as pd


def export_metrics(metrics):
    with open("outputs/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)


def export_predictions(predictions):
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


def export_trajectory(y_test, predictions):
    trajectory = []

    for i in range(len(predictions)):
        point = {
            "actual_lat": float(y_test.iloc[i, 0]),
            "actual_lon": float(y_test.iloc[i, 1]),
            "actual_alt": float(y_test.iloc[i, 2]),
            "predicted_lat": float(predictions[i, 0]),
            "predicted_lon": float(predictions[i, 1]),
            "predicted_alt": float(predictions[i, 2])
        }

        trajectory.append(point)

    with open("outputs/trajectory.json", "w") as file:
        json.dump(trajectory, file, indent=4)