import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy import stats


class AlphaInvestingSelector:
    def __init__(self, base_model=None, alpha_threshold=0.01, budget=0.5):
        """
        Alpha-investing feature selector following the paper's algorithm.
        
        Args:
            base_model: Base model for evaluation (default: LogisticRegression)
            alpha_threshold: alpha_delta - wealth increase when feature is added (default: 0.01)
            budget: initial_wealth W0 (default: 0.5)
        """
        self.selected_features = set()
        self.alpha_threshold = alpha_threshold  # This is alpha_delta in the paper
        self.budget = budget  # This is the wealth w_i
        self.model = base_model or LogisticRegression(max_iter=1000)
        self.feature_index = 1
        self.processed_features = set()

    def update(self, X, y):
        """
        Update the feature selection with new data, processing features sequentially.
        """
        if hasattr(X, 'empty') and X.empty:
            return
        if len(X) < 2:
            return

        if hasattr(X, 'columns'):
            all_features = list(X.columns)
        else:
            all_features = list(range(X.shape[1]))
        
        new_features = [f for f in all_features if f not in self.processed_features]
        
        for feature in new_features:
            if self.budget <= 0:
                break
                
            alpha_i = self.budget / (2 * self.feature_index)
            
            p_value = self._get_p_value(feature, X, y)
            
            if p_value < alpha_i:
                self.selected_features.add(feature)
                self.budget = self.budget + self.alpha_threshold - alpha_i
            else:
                self.budget = self.budget - alpha_i
            
            self.processed_features.add(feature)
            self.feature_index += 1
            
            self.budget = max(0, self.budget)

    def _get_p_value(self, feature, X, y):
        """
        Calculate p-value for adding a feature to the current model.
        Uses likelihood ratio test between current model and model with new feature.
        """
        try:
            if len(self.selected_features) == 0:
                null_model = LogisticRegression(fit_intercept=True, max_iter=1000)
                null_model.fit(np.ones((len(y), 1)), y)
                null_pred = null_model.predict_proba(np.ones((len(y), 1)))
                null_log_likelihood = -log_loss(y, null_pred)
            else:
                current_features = self._get_feature_data(list(self.selected_features), X)
                null_model = LogisticRegression(fit_intercept=True, max_iter=1000)
                null_model.fit(current_features, y)
                null_pred = null_model.predict_proba(current_features)
                null_log_likelihood = -log_loss(y, null_pred)
            
            new_features_list = list(self.selected_features) + [feature]
            new_features = self._get_feature_data(new_features_list, X)
            
            new_model = LogisticRegression(fit_intercept=True, max_iter=1000)
            new_model.fit(new_features, y)
            new_pred = new_model.predict_proba(new_features)
            new_log_likelihood = -log_loss(y, new_pred)
            
            lr_statistic = -2 * (null_log_likelihood - new_log_likelihood)
            
            lr_statistic = max(0, lr_statistic)
            
            p_value = 1 - stats.chi2.cdf(lr_statistic, df=1)
            
            return max(p_value, 1e-16)
            
        except Exception as e:
            return 1.0
    
    def _get_feature_data(self, feature_list, X):
        """Helper to extract feature data based on feature list."""
        if hasattr(X, 'iloc'):
            if isinstance(feature_list[0], str):
                return X[feature_list].values
            else:
                return X.iloc[:, feature_list].values
        else:  # numpy array
            return X[:, feature_list]

    def get_selected_features(self):
        """Return list of selected features."""
        return list(self.selected_features)
