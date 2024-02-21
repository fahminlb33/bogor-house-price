import matplotlib.pyplot as plt


def plot_predictions(y_test, y_pred):
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.axis("tight")

    return fig


def plot_residuals(y_test, y_pred):
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_test - y_pred, color="blue", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.axis("tight")

    return fig


def plot_distributions(y_test, y_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.hist(y_test, bins=50)
    ax1.set_title("Actual")

    ax2.hist(y_pred, bins=50)
    ax2.set_title("Predicted")

    return fig
