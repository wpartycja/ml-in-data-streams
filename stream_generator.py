"""Generates a stream of data with changing feature importances."""

from river import datasets
from scipy.special import expit
import numpy as np
from typing import Literal
import copy


class FeatureImportanceChangeGenerator(datasets.base.SyntheticDataset):
    def __init__(
        self,
        n_features: int,
        n_important_features: int,
        drift_type: Literal['abrupt', 'incremental', 'mixed'] = 'mixed',
        feature_drift_proba: float = 0,
        importance_change_interval: int = 3,
        random_seed: int = 2137,
        wrong_label_proba: float = 0.01,
        n_features_to_keep: int = None,
        add_noise: bool = True
    ):
        """
        Generates a stream of data with changing feature importances.
        The stream is generated by a linear model with a fixed number of features.
        The weights of the features are generated from a normal distribution.
        The important features are chosen randomly from the set of features.
        The important features are the ones that have a non-zero weight.
        The weights of the important features are generated from a normal distribution.
        The weights of the non-important features are set to zero (or a small value if add_noise=True).
        """
        super().__init__(
            task=datasets.base.BINARY_CLF,
            n_features=n_features,
            n_classes=2,
            n_outputs=1,
        )
        self.importance_change_interval = importance_change_interval
        self.n_important_features = n_important_features
        self.n_features_to_keep = n_features_to_keep if n_features_to_keep else n_important_features // 2
        self.feature_drift_proba = feature_drift_proba if feature_drift_proba else 0
        self.feature_drift_generator = np.random.default_rng(random_seed)
        self.change_interval_generator = np.random.default_rng(random_seed)
        self.iter = 0
        self.importance_history = {}
        self.feature_generators = [
            np.random.default_rng(random_seed + i) for i in range(n_features)
        ]
        self.weights_generator = np.random.default_rng(random_seed * 2)
        self.iters_left_to_change = 0
        self.weights = None
        self.old_weights = None
        self.important_features = None
        self.random_seed = random_seed
        self.add_noise = add_noise
        self.features = {i:[0, 1] for i in range(n_features)}
        self.feature_drifts = {}

        self.increment = None
        self.iters_left_of_increment = 0

        self.drift_type_generator = np.random.default_rng(random_seed)
        self.drift_type = drift_type
        self.next_drift_type = 'abrupt' if drift_type == 'abrupt' else \
                                ('incremental' if drift_type == 'incremental' else \
                                 ('abrupt' if self.drift_type_generator.random() >= 0.5 else 'incremental'))
        self.current_incremental = False
        self.wrong_label_proba = wrong_label_proba
        self.wrong_label_generator = np.random.default_rng(random_seed // 3)

    def __iter__(self):
        while True:
            self.iter += 1
            if self.iters_left_to_change == 0:
                self.iters_left_to_change = self.change_interval_generator.poisson(
                    self.importance_change_interval
                ) * 100 + np.round(self.change_interval_generator.normal(20, 50))
                if self.next_drift_type == 'abrupt':
                    self.change_weights_abrupt()
                elif self.next_drift_type == 'incremental':
                    self.change_weights_incremental()
                    self.current_incremental = True
                self.next_drift_type = 'abrupt' if self.drift_type == 'abrupt' else \
                                        ('incremental' if self.drift_type == 'incremental' else \
                                        ('abrupt' if self.drift_type_generator.random() >= 0.5 else 'incremental'))

            if self.current_incremental:
                self.iters_left_of_increment -= 1
                if self.iters_left_of_increment == 0:
                    self.current_incremental = False
                self.weights += self.increment
            if self.feature_drift_generator.random() < self.feature_drift_proba:
                feature_to_drift = self.feature_drift_generator.choice(range(self.n_features), 1).item()
                self.features[feature_to_drift][0] += np.round(self.feature_drift_generator.normal(0, 1)).item()
                self.features[feature_to_drift][1] += np.round(self.feature_drift_generator.normal(0, 1)).item()
                self.features[feature_to_drift][1] = np.abs(self.features[feature_to_drift][1]).item()
                self.feature_drifts[self.iter] = (feature_to_drift,
                                                  copy.deepcopy(self.features))
            features = [gen.normal(self.features[i][0], self.features[i][1]) for i, gen in zip(range(self.n_features), self.feature_generators)]
            x = dict(zip([f"var_{i}" for i in range(self.n_features)], features))
            probas = expit(features @ self.weights)
            y = self.change_interval_generator.binomial(1, probas)
            if self.wrong_label_generator.random() <= self.wrong_label_proba:
                y = 1 - y #0 -> 1, 1 -> 0
            self.iters_left_to_change -= 1
            yield x, y

    def change_weights_abrupt(self):
        """Change the weights of the features in an abrupt manner."""
        self.get_new_weights()
        self.importance_history[self.iter] = (self.important_features, self.weights.copy(), 'abrupt')

    def change_weights_incremental(self):
        """Change the weights of the features in an incremental manner."""
        self.old_weights = self.weights
        self.get_new_weights()
        self.importance_history[self.iter] = (self.important_features, self.weights.copy(), 'incremental')
        self.iters_left_of_increment = self.iters_left_to_change // 5
        self.increment = (self.weights - self.old_weights) / self.iters_left_of_increment
        self.weights = self.old_weights

    def get_new_weights(self):
        """Generate new weights"""
        self.weights = np.diagonal(
            self.weights_generator.normal(4, 1, size=(self.n_features, 1))
            @ self.weights_generator.normal(5, 2, size=(1, self.n_features))
        )
        if self.important_features is not None:
            old_features = self.important_features
            features_to_keep = self.change_interval_generator.choice(old_features,
                                                                    self.n_features_to_keep,
                                                                    replace=False
                                                                    )
            features_other = list(set(range(self.n_features)) - set(features_to_keep))
            self.important_features = list(self.change_interval_generator.choice(
                features_other,
                self.n_important_features - self.n_features_to_keep,
                replace=False
            )) + list(features_to_keep)
        else:
            self.important_features = self.change_interval_generator.choice(
            self.n_features, self.n_important_features, replace=False
        )
        self.important_features = \
            [x.item() for x in self.important_features if isinstance(x, np.int64)]
        self.important_features.sort()
        mask = np.zeros_like(self.weights)
        mask[self.important_features] = 1
        self.weights = mask * self.weights
        if self.add_noise:
            noise = self.weights_generator.normal(3, 1, size=self.weights.shape)
            self.weights += noise
