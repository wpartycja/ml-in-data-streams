import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class AlphaInvestingSelector:
    def __init__(self, base_model=None, alpha_threshold=0.01):
        self.selected_features = set()
        self.alpha_threshold = alpha_threshold
        self.budget = 0.1
        self.model = base_model or LogisticRegression()

    def update(self, X, y):
        if X.empty or len(X) < 50:
            return  # not enough data to evaluate anything

        # If first time, add top correlated feature
        if not self.selected_features:
            corr = X.apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))
            best_feat = corr.idxmax()
            self.selected_features.add(best_feat)
            return

        baseline_perf = self._evaluate(list(self.selected_features), X, y)

        for f in X.columns:
            if f in self.selected_features:
                continue

            candidate_feats = list(self.selected_features | {f})
            new_perf = self._evaluate(candidate_feats, X, y)

            if (new_perf - baseline_perf) > self.alpha_threshold:
                self.selected_features.add(f)
                self.budget += 0.05
            else:
                self.budget -= 0.01

            if self.budget <= 0:
                break

    def _evaluate(self, features, X, y):
        try:
            self.model.fit(X[features], y)
            pred = self.model.predict_proba(X[features])
            return -log_loss(y, pred, labels=np.unique(y))
        except Exception:
            return float('-inf')

    def get_selected_features(self):
        return list(self.selected_features)

