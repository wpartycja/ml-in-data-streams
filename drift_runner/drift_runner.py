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
from drift_runner.models import NoChangeModel, MajorityClassModel

class DriftDetectionRunner:
    def __init__(self, generator, drift_detector, sensitive_drift_detector, model_type='hoeffding', feature_selector=None, initial_samples=100, n_samples=10000, print_warning=False, plot_path=None, export_path=None, print_plot=True):
        self.generator = generator
        self.drift_detector = drift_detector
        self.sensitive_drift_detector = sensitive_drift_detector
        self.feature_selector_type = feature_selector
        self.alpha_selector = AlphaInvestingSelector() if feature_selector == 'alpha' else None
        self.initial_samples = initial_samples
        self.n_samples = n_samples
        self.print_warning = print_warning
        self.plot_path = plot_path
        self.export_path = export_path
        self.print_plot = print_plot
        
        self.model_type = model_type
        self._init_model()
        
        self.warning_data_xs = deque()
        self.warning_data_ys = deque()
        self.collecting_warning_data = False

        self.acc = metrics.Accuracy()
        self.report = metrics.ClassificationReport()

        self.accuracies = []
        self.detected_drift_points = []
        self.regular_model_changes = []
        self.epochs = []
        self.sensitive_drift_points = []

        self.previous_xs = deque()
        self.previous_ys = deque()

    def _init_model(self):
        if self.model_type == 'hoeffding':
            self.model = tree.HoeffdingAdaptiveTreeClassifier()
        elif self.model_type == 'majority':
            self.model = MajorityClassModel()
        elif self.model_type == 'no_change':
            self.model = NoChangeModel()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def initialize_history(self):
        for x, y in self.generator.take(self.initial_samples):
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

        if self.sensitive_drift_detector:
            self.sensitive_drift_detector.update(y == y_pred)
            if self.sensitive_drift_detector.drift_detected and not self.collecting_warning_data:
                print(f"[Sensitive] Simulated warning: sensitive drift detector detected early drift at epoch {epoch}")
                self.collecting_warning_data = True
                self.warning_data_xs.clear()
                self.warning_data_ys.clear()
                self.sensitive_drift_points.append(epoch)

        drift_detected = self.drift_detector.drift_detected
        if drift_detected:
            print(f"[Main] Drift detected at index {epoch}")
            if self.collecting_warning_data:
                print(f"[Sensitive] Finalizing warning-phase collection. Epoch {epoch}")
                self.previous_xs.extend(self.warning_data_xs)
                self.previous_ys.extend(self.warning_data_ys)
            self.collecting_warning_data = False
            return True

        return False


    def _get_new_features(self):
        df_x = pd.DataFrame(self.previous_xs)
        arr_y = np.array(self.previous_ys)

        if self.feature_selector_type == 'boruta':
            boruta_result = run_Boruta(df_x, arr_y)
            return list(boruta_result.accepted) + list(boruta_result.tentative)

        elif self.feature_selector_type == 'alpha':
            self.alpha_selector = AlphaInvestingSelector()
            self.alpha_selector.update(df_x, arr_y)
            return self.alpha_selector.get_selected_features()

        elif self.feature_selector_type is None:
            return list(df_x.columns)  # Use all features

        else:
            raise ValueError(f"Unknown feature selector type: {self.feature_selector_type}")

    def _run(self):
        for epoch, (x, y) in enumerate(self.generator.take(self.n_samples - self.initial_samples)):
            self.iter = epoch  # <-- Add this line to keep track of stream position

            self.previous_xs.append(x)
            self.previous_ys.append(y)

            if self.collecting_warning_data:
                self.warning_data_xs.append(x)
                self.warning_data_ys.append(y)

            important_xs = self.get_features(x)
            
            y_pred = self._update_metrics(important_xs, y, epoch)

            if self.drift_detector is None and self.model_type == 'hoeffding':
                if epoch+self.initial_samples in self.generator.importance_history:
                    self.handle_drift(epoch+self.initial_samples)

            elif y_pred is not None and self._check_drift(y, y_pred, epoch):
                self.handle_drift(epoch)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, {self.acc}")

        self._finalize()
    
    def handle_drift(self, epoch):
        self.detected_drift_points.append(epoch)
        self.accepted_features = self._get_features_from_warning_data()
        self._init_model()
        self.warning_data_xs.clear()
        self.warning_data_ys.clear()
        print(f"Drift handled at epoch {epoch}. Model and features updated.")
    
    def _get_features_from_warning_data(self):
        if not self.warning_data_xs:
            print("[Warning] No warning data available for feature selection.")
            return self.accepted_features  # fallback to last known features

        df_x = pd.DataFrame(self.warning_data_xs)
        arr_y = np.array(self.warning_data_ys)

        if self.feature_selector_type == 'boruta':
            boruta_result = run_Boruta(df_x, arr_y)
            return list(boruta_result.accepted) + list(boruta_result.tentative)

        elif self.feature_selector_type == 'alpha':
            self.alpha_selector.update(df_x, arr_y)
            return self.alpha_selector.get_selected_features()

        else:
            raise ValueError(f"Unknown feature selector type: {self.feature_selector_type}")

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
            self.print_plot,
            self.sensitive_drift_points 
        )
        
        if self.export_path:
            self.export_run_data(self.export_path)

    def run(self, mode='boruta_dynamic'):
        """
        Supported Modes:
        - 'all_features_no_reset': Use all features, no action on drift.
        - 'all_features_with_reset': Use all features, reset model on drift.
        - 'boruta_dynamic': Sensitive + main drift detection, use Boruta for feature selection on drift.
        - 'alpha_dynamic': Use AlphaInvesting for online feature selection; reset model and update features on drift.
        """

        # Override mode if using a baseline model
        if self.model_type in ['majority', 'no_change']:
            print(f"[Info] Overriding mode to 'all_features_no_reset' for baseline model '{self.model_type}'")
            mode = 'all_features_no_reset'
        
        # Force mode to alpha_dynamic if using AlphaInvesting
        if self.feature_selector_type == 'alpha' and mode != 'alpha_dynamic':
            raise ValueError(
                f"[Error] feature_selector='alpha' requires mode='alpha_dynamic', got mode='{mode}'"
        )
            
        if self.feature_selector_type == 'boruta' and mode == 'alpha_dynamic':
            raise ValueError(
                f"[Error] Cannot use feature_selector='boruta' with mode='alpha_dynamic'"
            )
        
        if self.feature_selector_type is not None and mode == 'oracle':
            raise ValueError(
                f"[Error] Cannot use feature_selector with mode='oracle'"
            )

        self.initialize_history()

        mode_decorators = {
            'all_features_no_reset': all_features_no_reset,
            'all_features_with_reset': all_features_with_reset,
            'boruta_initial_only': boruta_initial_only,
            'boruta_dynamic': boruta_dynamic,
            'alpha_dynamic': alpha_dynamic,
            'oracle': oracle
        }

        if mode not in mode_decorators:
            raise ValueError(f"Unknown mode '{mode}', expected one of {list(mode_decorators)}")

        decorated_run = mode_decorators[mode](type(self)._run).__get__(self, type(self))
        decorated_run()

    def export_run_data(self, filepath):
        detector_config = {
            "feature_selector": self.feature_selector_type,
            "drift_detector": type(self.drift_detector).__name__,
            "boruta_samples": self.initial_samples,
            "n_samples": self.n_samples
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
