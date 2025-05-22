from matplotlib import pyplot as plt
import os
import json 
import matplotlib.cm as cm


def plot_accuracy_with_drift(
    epochs, accuracies, real_drift_points, detected_drift_points, regular_change_points,
    save_path=None, print_plot=True, sensitive_drift_points=None
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
    
    if sensitive_drift_points:
        for i, drift in enumerate(sensitive_drift_points):
            plt.axvline(
                x=drift,
                color="orange",
                linestyle="dotted",
                label="Sensitive Drift" if i == 0 else "",
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


def plot_all_strategies(detector, seed, window, strategies):

    plt.figure(figsize=(14, 6))
    real_drift_lines_added = False
    
    
    # Generate a distinct color for each strategy
    colors = cm.get_cmap('prism', len(strategies))  # 'Tab1', 'Set1'
    
    for col, strategy in enumerate(strategies):
        # Choose path based on strategy type
        if strategy in ['boruta_initial_only', 'boruta_dynamic', 'alpha_dynamic']:
            path = f"logs/{detector}/{seed}_seed/{window}_window/{strategy}.json"
        else:
            path = f"logs/{detector}/{seed}_seed/{strategy}.json"

        with open(path, "r") as f:
            data = json.load(f)

        results = data["results"]
        epochs = results["epochs"]
        accuracies = results["accuracies"]
        real_drifts = results["real_drift_points"]
        detected_drifts = results["detected_drift_points"]
        model_changes = results.get("regular_model_changes", [])

        color = colors(col)

        # Plot accuracy line
        plt.plot(epochs, accuracies, label=strategy, color=color)


    # Labels and legend
    plt.title(f"Accuracy Over Time for Different Feature Selection Strategies \n Detector: {detector}, Window: {window}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'./plots_all_strategies/{detector}_{seed}seed_{window}window')
    plt.show()
