from feature_selection.boruta import run_Boruta
import numpy as np
import pandas as pd
from drift_runner.utils import plot_accuracy_with_drift
from river import tree, metrics
from collections import deque
from drift_runner.drift_strategies import *
from feature_selection.alpha_investing import AlphaInvestingSelector
import json
import os

class DriftDetectionRunner:
    def __init__(self, generator, drift_detector,  feature_selector='boruta', boruta_samples=100, n_samples=10000, n_history=1000, print_warning=False, plot_path=None, export_path=None, print_plot=True):
        self.generator = generator
        self.drift_detector = drift_detector
        self.feature_selector_type = feature_selector
        self.alpha_selector = AlphaInvestingSelector() if feature_selector == 'alpha' else None
        self.boruta_samples = boruta_samples
        self.n_samples = n_samples
        self.n_history = n_history
        self.print_warning = print_warning
        self.plot_path = plot_path
        self.export_path = export_path
        self.print_plot = print_plot

        self.model = tree.HoeffdingAdaptiveTreeClassifier()
        self.acc = metrics.Accuracy()
        self.report = metrics.ClassificationReport()

        self.accuracies = []
        self.detected_drift_points = []
        self.regular_model_changes = []
        self.epochs = []

        self.previous_xs = deque(maxlen=n_history)
        self.previous_ys = deque(maxlen=n_history)

    def initialize_history(self):
        for x, y in self.generator.take(self.boruta_samples):
            self.previous_xs.append(x)
            self.previous_ys.append(y)

    def get_features(self, x):
        if not self.accepted_features:
            return x  # fallback to all features
        return {k: v for k, v in x.items() if k in self.accepted_features}
        
    def _update_metrics(self, x, y, epoch):
        y_pred = self.model.predict_one(x)
        self.model.learn_one(x, y)

        if y_pred is not None:
            self.report.update(y, y_pred)
            self.acc.update(y, y_pred)
            self.accuracies.append(self.acc.get())
            self.epochs.append(epoch)

        return y_pred


    def _check_drift(self, y, y_pred, epoch):
        self.drift_detector.update(y == y_pred)
        drift_detected = self.drift_detector.drift_detected
        warning_detected = getattr(self.drift_detector, "warning_detected", False)

        if self.print_warning and warning_detected:
            print(f"Warning detected at index {epoch}, input value: {y}")

        if drift_detected:
            print(f"Drift detected at index {epoch}, input value: {y}")

        return drift_detected


    def _get_new_features(self):
        df_x = pd.DataFrame(self.previous_xs)
        arr_y = np.array(self.previous_ys)

        if self.feature_selector_type == 'boruta':
            boruta_result = run_Boruta(df_x, arr_y)
            return list(boruta_result.accepted) + list(boruta_result.tentative)

        elif self.feature_selector_type == 'alpha':
            self.alpha_selector.update(df_x, arr_y)
            return self.alpha_selector.get_selected_features()

        else:
            raise ValueError(f"Unknown feature selector type: {self.feature_selector_type}")

    def _run(self):
        for epoch, (x, y) in enumerate(self.generator.take(self.n_samples - self.boruta_samples)):
            self.previous_xs.append(x)
            self.previous_ys.append(y)

            important_xs = self.get_features(x)
            y_pred = self._update_metrics(important_xs, y, epoch)

            if y_pred is not None and self._check_drift(y, y_pred, epoch):
                self.handle_drift(epoch)

            # Optional: track feature stability for mode D @TODO ????????????????????
            elif self.accepted_features is not None and epoch % self.n_history == 0:
                new_features = self._get_new_features()
                if len(set(self.accepted_features).difference(new_features)) > len(self.accepted_features) / 2:
                    print(f"Feature importance changed at epoch {epoch}.")
                    self.accepted_features = new_features
                    self.model = tree.HoeffdingTreeClassifier()
                    self.regular_model_changes.append(epoch)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, {self.acc}")

        self._finalize()

    def _finalize(self):
        print('\n')
        print(self.report)
        real_drift_points = np.array(list(self.generator.importance_history.keys())) - 100
        plot_accuracy_with_drift(
            self.epochs,
            self.accuracies,
            real_drift_points,
            self.detected_drift_points,
            self.regular_model_changes,
            self.plot_path,
            self.print_plot
        )
        
        if self.export_path:
            self.export_run_data(self.export_path)

    def run(self, mode='boruta_dynamic'):
        self.initialize_history()

        mode_decorators = {
            'all_features_no_reset': all_features_no_reset,
            'all_features_with_reset': all_features_with_reset,
            'boruta_initial_only': boruta_initial_only,
            'boruta_dynamic': boruta_dynamic,
            'alpha_dynamic': alpha_dynamic
        }

        if mode not in mode_decorators:
            raise ValueError(f"Unknown mode '{mode}', expected one of {list(mode_decorators)}")

        decorated_run = mode_decorators[mode](type(self)._run).__get__(self, type(self))
        decorated_run()


    def export_run_data(self, filepath):
        detector_config = {
            "feature_selector": self.feature_selector_type,
            "drift_detector": type(self.drift_detector).__name__,
            "boruta_samples": self.boruta_samples,
            "n_samples": self.n_samples,
            "n_history": self.n_history
        }

        generator_config = {
            "n_features": getattr(self.generator, 'n_features', None),
            "n_important_features": getattr(self.generator, 'n_important_features', None),
            "importance_change_interval": getattr(self.generator, 'importance_change_interval', None)
        }

        results_data = {
            "epochs": self.epochs,
            "accuracies": self.accuracies,
            "real_drift_points": (np.array(list(self.generator.importance_history.keys())) - 100).tolist(),
            "detected_drift_points": self.detected_drift_points,
            "regular_model_changes": self.regular_model_changes
        }

        data = {
            "detector_runner": detector_config,
            "generator": generator_config,
            "results": results_data
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Run data exported to {filepath}")
