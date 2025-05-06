from matplotlib import pyplot as plt


def plot_accuracy_with_drift(
    epochs, accuracies, real_drift_points, detected_drift_points, regular_change_points, save_path=None
):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracies, label="Accuracy", color="blue")

    for i, drift in enumerate(real_drift_points):
        plt.axvline(
            x=drift,
            color="violet",
            linestyle="solid",
            label="Real Drift Occured" if i == 0 else "",
        )
    for i, drift in enumerate(detected_drift_points):
        plt.axvline(
            x=drift,
            color="indigo",
            linestyle="dashed",
            label="Drift Detected" if i == 0 else "",
        )
    for i, drift in enumerate(regular_change_points):
        plt.axvline(
            x=drift,
            color="lightgreen",
            linestyle="dashed",
            label="Drift Detected (Regular)" if i == 0 else "",
        )

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Time with Drift Detection")
    plt.legend()
    
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
    plt.show()
    plt.close()
