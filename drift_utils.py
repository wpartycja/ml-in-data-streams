from feature_selection import run_Boruta
import numpy as np
import pandas as pd
from plots import plot_accuracy_with_drift
from river import tree, metrics
from collections import deque

def update_metrics(model, x, y, acc, classification_report, epoch, accuracies, epochs):
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    if y_pred is not None:
        classification_report.update(y, y_pred)
        acc.update(y, y_pred)
        accuracies.append(acc.get())
        epochs.append(epoch)
    return model, y_pred


def check_drift(drift_detector, y, y_pred, epoch, print_warning):
    drift_detector.update(y == y_pred)
    drift_detected = drift_detector.drift_detected
    warning_detected = getattr(drift_detector, "warning_detected", False)

    if print_warning and warning_detected:
        print(f"Warning detected at index {epoch}, input value: {y}")

    if drift_detected:
        print(f"Drift detected at index {epoch}, input value: {y}")
    return drift_detected


def get_new_boruta_features(previous_xs, previous_ys):
    boruta_result = run_Boruta(pd.DataFrame(previous_xs), np.array(previous_ys))
    accepted_features = list(boruta_result.accepted)
    accepted_features.extend(list(boruta_result.tentative))
    
    return accepted_features


def run_drift_detection(generator, drift_detector, boruta_samples=100, n_samples=10_000, n_history=1000, print_warning=False):
    
    acc = metrics.Accuracy()
    # model = tree.HoeffdingTreeClassifier()
    model = tree.HoeffdingAdaptiveTreeClassifier()
    classification_report = metrics.ClassificationReport()

    accuracies, detected_drift_points, regular_model_changes, epochs = [], [], [], []
    previous_xs, previous_ys = deque(maxlen=n_history), deque(maxlen=n_history)
    
    # get first x samples to find buruta features
    for x, y in generator.take(boruta_samples):
        previous_xs.append(x), previous_ys.append(y)

    accepted_features = get_new_boruta_features(previous_xs, previous_ys)
    

    # start stream
    for epoch, (x, y) in enumerate(generator.take(n_samples-boruta_samples)):
        # update history
        previous_xs.append(x), previous_ys.append(y)
        
        # single element prediction
        important_xs = {key: value for key, value in x.items() if key in accepted_features}
        model, y_pred = update_metrics(model, important_xs, y, acc, classification_report, epoch, accuracies, epochs)

        # check drift on y
        if y_pred is not None and check_drift(drift_detector, y, y_pred, epoch, print_warning):
            accepted_features = get_new_boruta_features(previous_xs, previous_ys)

            model = tree.HoeffdingTreeClassifier()
            detected_drift_points.append(epoch)
        
        # check feature importance every x epochs:
        elif epoch % n_history == 0:
            
            new_accepted_features = get_new_boruta_features(previous_xs, previous_ys)

            if len(set(accepted_features).difference(new_accepted_features)) > len(accepted_features) / 2:
                print(f"Feature importance changed at epoch {epoch}.")
                accepted_features = new_accepted_features
                model = tree.HoeffdingTreeClassifier()
                regular_model_changes.append(epoch)
        
        if epoch % 1_000 == 0:
            print(f'Epoch {epoch}, {acc}')

    print('\n')
    print(classification_report)
    
    real_drfit_points = np.array(list((generator.importance_history.keys()))) - 100
    
    plot_accuracy_with_drift(epochs, accuracies, real_drfit_points, detected_drift_points, regular_model_changes)