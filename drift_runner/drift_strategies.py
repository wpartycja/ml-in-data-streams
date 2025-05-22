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
        self.accepted_features = self._get_new_features()
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
        self.accepted_features = self._get_new_features()

        def reset_and_update(epoch):
            print(f"Drift detected at epoch {epoch} → updating features and resetting model")
            self.accepted_features = self._get_new_features()
            self.model = tree.HoeffdingTreeClassifier()
            self.detected_drift_points.append(epoch)

        self.handle_drift = reset_and_update
        return func(self, *args, **kwargs)
    return wrapper


def alpha_dynamic(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.accepted_features = self._get_new_features()

        def reset_model(epoch):
            print(f"Drift detected at epoch {epoch} → resetting model (Alpha mode)")
            self.model = tree.HoeffdingTreeClassifier()
            self.detected_drift_points.append(epoch)
            # Update alpha-selected features again after reset
            self.accepted_features = self._get_new_features()

        self.handle_drift = reset_model
        return func(self, *args, **kwargs)
    return wrapper


def oracle(func):
    def wrapper(self, *args, **kwargs):
        def oracle_update(epoch):
            print(f"[Oracle] Drift at epoch {epoch} → resetting model and applying oracle features")
            valid_points = [t for t in self.generator.importance_history if t <= epoch]
            if not valid_points:
                earliest = min(self.generator.importance_history)
                print(f"[Oracle] No drift <= {epoch}, falling back to earliest: {earliest}")
                current = earliest
            else:
                current = max(valid_points)

            important_features, _ = self.generator.importance_history[current]
            self.accepted_features = [f"var_{i}" for i in important_features]
            self._init_model()
            self.detected_drift_points.append(epoch)

        self.handle_drift = oracle_update

        if self.generator.importance_history:
            first = min(self.generator.importance_history)
            oracle_update(first)
        else:
            raise ValueError("importance_history is empty. Generator hasn't recorded any feature changes yet.")

        return func(self, *args, **kwargs)
    return wrapper
