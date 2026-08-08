import matplotlib.pyplot as plt
import numpy as np


EARTH_RADIUS_M = 6371000


def plot_feature_importance(model):

    importance = np.mean(
        [
            estimator.feature_importances_
            for estimator in model.estimators_
        ],
        axis=0
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        range(len(importance)),
        importance
    )

    plt.title(
        "Feature Importance"
    )

    plt.xlabel(
        "Feature Index"
    )

    plt.ylabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/feature_importance.png"
    )

    plt.close()


def plot_latitude_error(
    y_test,
    predictions
):

    error = (
        y_test.iloc[:, 0].values
        - predictions[:, 0]
    )

    plt.figure(figsize=(10, 5))

    plt.scatter(
        range(len(error)),
        error,
        s=10
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.title(
        "Latitude Error"
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Error (degrees)"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/latitude_error.png"
    )

    plt.close()


def plot_longitude_error(
    y_test,
    predictions
):

    error = (
        y_test.iloc[:, 1].values
        - predictions[:, 1]
    )

    plt.figure(figsize=(10, 5))

    plt.scatter(
        range(len(error)),
        error,
        s=10
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.title(
        "Longitude Error"
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Error (degrees)"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/longitude_error.png"
    )

    plt.close()


def plot_altitude_error(
    y_test,
    predictions
):

    error = (
        y_test.iloc[:, 2].values
        - predictions[:, 2]
    )

    plt.figure(figsize=(10, 5))

    plt.scatter(
        range(len(error)),
        error,
        s=10
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.title(
        "Altitude Error"
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Error (m)"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/altitude_error.png"
    )

    plt.close()


def calculate_3d_position_error(
    y_actual,
    predictions
):

    actual_lat = y_actual.iloc[:, 0].values
    actual_lon = y_actual.iloc[:, 1].values
    actual_alt = y_actual.iloc[:, 2].values

    predicted_lat = predictions[:, 0]
    predicted_lon = predictions[:, 1]
    predicted_alt = predictions[:, 2]

    lat_error_deg = (
        actual_lat - predicted_lat
    )

    lon_error_deg = (
        actual_lon - predicted_lon
    )

    alt_error_m = (
        actual_alt - predicted_alt
    )

    mean_lat_rad = np.radians(
        (actual_lat + predicted_lat) / 2
    )

    lat_error_m = (
        np.radians(lat_error_deg)
        * EARTH_RADIUS_M
    )

    lon_error_m = (
        np.radians(lon_error_deg)
        * EARTH_RADIUS_M
        * np.cos(mean_lat_rad)
    )

    position_error_3d_m = np.sqrt(
        lat_error_m ** 2
        + lon_error_m ** 2
        + alt_error_m ** 2
    )

    return (
        lat_error_m,
        lon_error_m,
        alt_error_m,
        position_error_3d_m
    )


def plot_3d_position_error(
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

    time_values = np.asarray(
        time_all
    )

    order = np.argsort(
        time_values
    )

    time_values = time_values[order]

    position_error_3d_m = (
        position_error_3d_m[order]
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        time_values,
        position_error_3d_m,
        linewidth=1
    )

    plt.title(
        "3D Position Error vs Time"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Position Error (m)"
    )

    plt.xlim(
        time_values.min(),
        time_values.max()
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/3d_position_error.png",
        dpi=150
    )

    plt.close()


def plot_residuals(
    y_test,
    predictions
):

    residuals = np.ravel(
        y_test.values - predictions
    )

    plt.figure(figsize=(10, 5))

    plt.hist(
        residuals,
        bins=30
    )

    plt.title(
        "Residual Distribution"
    )

    plt.xlabel(
        "Residual"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/residuals.png"
    )

    plt.close()


def plot_trajectory(
    y_test,
    predictions
):

    plt.figure(
        figsize=(8, 8)
    )

    plt.scatter(
        y_test.iloc[:, 0],
        y_test.iloc[:, 1],
        s=10,
        label="Actual"
    )

    plt.scatter(
        predictions[:, 0],
        predictions[:, 1],
        s=10,
        label="Predicted"
    )

    plt.xlabel(
        "Latitude"
    )

    plt.ylabel(
        "Longitude"
    )

    plt.title(
        "Trajectory"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/trajectory.png"
    )

    plt.close()


def plot_actual_vs_predicted(
    y_test,
    predictions
):

    plt.figure(
        figsize=(8, 8)
    )

    plt.scatter(
        y_test.iloc[:, 0],
        predictions[:, 0],
        s=10
    )

    minimum = min(
        y_test.iloc[:, 0].min(),
        predictions[:, 0].min()
    )

    maximum = max(
        y_test.iloc[:, 0].max(),
        predictions[:, 0].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel(
        "Actual Latitude"
    )

    plt.ylabel(
        "Predicted Latitude"
    )

    plt.title(
        "Actual vs Predicted Latitude"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/actual_vs_predicted.png"
    )

    plt.close()