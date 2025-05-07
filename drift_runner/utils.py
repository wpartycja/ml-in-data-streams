from matplotlib import pyplot as plt
import os


def plot_accuracy_with_drift(
    epochs, accuracies, real_drift_points, detected_drift_points, regular_change_points, save_path=None, print_plot=True
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
    if print_plot: 
        plt.show()
    plt.close()


def get_paths(chosen_detector, seed, n_history, strategy):
    if strategy in ['boruta_initial_only', 'boruta_dynamic', 'alpha_dynamic']:
        plot_path = f'./plots/{chosen_detector}/{seed}_seed/{n_history}_window/{strategy}.png'
        export_path = f'./logs/{chosen_detector}/{seed}_seed/{n_history}_window/{strategy}.json'
    else:
        plot_path = f'./plots/{chosen_detector}/{seed}_seed/{strategy}.png'
        export_path = f'./logs/{chosen_detector}/{seed}_seed/{strategy}.json'
    
    return plot_path, export_path