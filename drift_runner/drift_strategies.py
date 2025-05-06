from functools import wraps
from river import tree

def all_features_no_reset(func):
    """All features, do nothing on drift"""
    def wrapper(self, *args, **kwargs):
        self.accepted_features = None  # use all features
        self.handle_drift = lambda epoch: None  # no action on drift
        return func(self, *args, **kwargs)
    return wrapper


def all_features_with_reset(func):
    """All features, reset model on drift"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.accepted_features = None
        def reset_model(epoch):
            print(f"Drift detected at epoch {epoch} → resetting model (mode B)")
            self.model = tree.HoeffdingTreeClassifier()
            self.detected_drift_points.append(epoch)
        self.handle_drift = reset_model
        return func(self, *args, **kwargs)
    return wrapper


def boruta_initial_only(func):
    """Fixed Boruta features, reset model on drift"""
    def wrapper(self, *args, **kwargs):
        self.accepted_features = self._get_new_boruta_features()
        def reset_model(epoch):
            print(f"Drift detected at epoch {epoch} → resetting model")
            self.model = tree.HoeffdingTreeClassifier()
            self.detected_drift_points.append(epoch)
        self.handle_drift = reset_model
        return func(self, *args, **kwargs)
    return wrapper


def boruta_dynamic(func):
    """Boruta features, update features + model on drift"""
    def wrapper(self, *args, **kwargs):
        self.accepted_features = self._get_new_boruta_features()

        def reset_and_update(epoch):
            print(f"Drift detected at epoch {epoch} → updating features and resetting model")
            self.accepted_features = self._get_new_boruta_features()
            self.model = tree.HoeffdingTreeClassifier()
            self.detected_drift_points.append(epoch)

        self.handle_drift = reset_and_update
        return func(self, *args, **kwargs)
    return wrapper
