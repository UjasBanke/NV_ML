import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance(model):
    estimator = model.estimators_[0]
    importance = estimator.feature_importances_

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importance)), importance)

    plt.title("Feature Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")

    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()


def plot_latitude_error(y_test, predictions):
    error = y_test.iloc[:, 0].values - predictions[:, 0]

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(error)), error, s=10)

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.title("Latitude Error")
    plt.xlabel("Sample")
    plt.ylabel("Error")

    plt.tight_layout()
    plt.savefig("outputs/latitude_error.png")
    plt.close()


def plot_longitude_error(y_test, predictions):
    error = y_test.iloc[:, 1].values - predictions[:, 1]

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(error)), error, s=10)

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.title("Longitude Error")
    plt.xlabel("Sample")
    plt.ylabel("Error")

    plt.tight_layout()
    plt.savefig("outputs/longitude_error.png")
    plt.close()


def plot_altitude_error(y_test, predictions):
    error = y_test.iloc[:, 2].values - predictions[:, 2]

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(error)), error, s=10)

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.title("Altitude Error")
    plt.xlabel("Sample")
    plt.ylabel("Error")

    plt.tight_layout()
    plt.savefig("outputs/altitude_error.png")
    plt.close()


def plot_residuals(y_test, predictions):
    residuals = np.ravel(y_test.values - predictions)

    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=30)

    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("outputs/residuals.png")
    plt.close()


def plot_trajectory(y_test, predictions):
    plt.figure(figsize=(8, 8))

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

    plt.xlabel("Latitude")
    plt.ylabel("Longitude")

    plt.title("Trajectory")

    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/trajectory.png")
    plt.close()


def plot_actual_vs_predicted(y_test, predictions):
    plt.figure(figsize=(8, 8))

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

    plt.xlabel("Actual Latitude")
    plt.ylabel("Predicted Latitude")
    plt.title("Actual vs Predicted")

    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png")
    plt.close()