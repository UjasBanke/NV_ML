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
    plt.savefig("outputs/feature_importance.png")
    plt.close()


def plot_latitude_error(y_test, predictions):
    error = y_test.iloc[:, 0] - predictions[:, 0]

    plt.figure(figsize=(10, 5))
    plt.plot(error)
    plt.title("Latitude Error")
    plt.savefig("outputs/latitude_error.png")
    plt.close()


def plot_longitude_error(y_test, predictions):
    error = y_test.iloc[:, 1] - predictions[:, 1]

    plt.figure(figsize=(10, 5))
    plt.plot(error)
    plt.title("Longitude Error")
    plt.savefig("outputs/longitude_error.png")
    plt.close()


def plot_altitude_error(y_test, predictions):
    error = y_test.iloc[:, 2] - predictions[:, 2]

    plt.figure(figsize=(10, 5))
    plt.plot(error)
    plt.title("Altitude Error")
    plt.savefig("outputs/altitude_error.png")
    plt.close()


def plot_residuals(y_test, predictions):
    residuals = np.ravel(y_test.values - predictions)

    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution")
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

    plt.legend()
    plt.title("Trajectory")
    plt.savefig("outputs/trajectory.png")
    plt.close()