import numpy as np
import pandas as pd
import scipy.stats as sps

from lib.ml import dist


class EMEmbedder:
    feature_type_to_dist_class = {
        'categ': dist.Categorical,
        'counter': dist.ZeroInflattedPoisson,
        'normal': dist.Normal,
        'log_normal': dist.Normal,
    }

    def __init__(self, embed_size, features, batch_size=None, max_iter=100):
        self.embed_size = embed_size
        self.features = features
        self.batch_size = batch_size
        self.max_iter = max_iter
        self._mappings = {}
        self._dists = {}
        self._data_to_cluster_affinity = None
        self._cluster_proba = None

    @staticmethod
    def _normalize_normal_sample(data):
        average, std = np.average(data), np.std(data)
        return np.clip((data - average) / (std + 0.1), -3, 3)

    def _normalize_feature(self, data, feature):
        if self.features[feature] == 'categ':
            if feature not in self._mappings:
                unique = np.unique(data)
                self._mappings[feature] = pd.Series(range(unique.shape[0]), index=unique)
            return self._mappings[feature][data].to_numpy()
        elif self.features[feature] == 'counter':
            return np.array(data, dtype=int)
        elif self.features[feature] == 'normal':
            return np.array(data, dtype=float)
        elif self.features[feature] == 'log_normal':
            return np.log(1. + np.array(data, dtype=float))
        else:
            raise NotImplementedError('Feature type "{}" not supported'.format(feature_type))

    def _normalize_data(self, data):
        normalized_data = pd.DataFrame({feature: self._normalize_feature(
            data[feature],
            feature,
        ) for feature in self.features})
        return normalized_data

    def _init_dists(self, data):
        self._cluster_proba = sps.dirichlet(np.ones(self.embed_size)).rvs()[0]
        for feature in self.features:
            self._dists[feature] = []
            for dim in range(self.embed_size):
                self._dists[feature].append(self.feature_type_to_dist_class[self.features[feature]]())
                self._dists[feature][-1].random_fit(data[feature])

    def _estimate_data_to_cluster_affinity(self, data):
        data_to_cluster_affinity = self._cluster_proba[None, :] * np.prod([[self._dists[feature][dim].proba(
            data[feature],
        ) for dim in range(self.embed_size)] for feature in self.features], axis=0).T
        return data_to_cluster_affinity \
            / data_to_cluster_affinity.sum(axis=1)[:, None]

    def _optimize_data_to_cluster_affinity(self, data):
        self._data_to_cluster_affinity = self._estimate_data_to_cluster_affinity(data)

    def _optimize_cluster_params(self, data):
        self._cluster_proba = self._data_to_cluster_affinity.mean(axis=0)
        for feature in self.features:
            for dim in range(self.embed_size):
                batch_data = data.sample(n=self.batch_size) if self.batch_size is not None else data
                self._dists[feature][dim].fit(data[feature], self._data_to_cluster_affinity[:, dim])

    def fit(self, data):
        data = self._normalize_data(data)
        self._init_dists(data)
        
        for it in range(self.max_iter):
            self._optimize_data_to_cluster_affinity(data)
            self._optimize_cluster_params(data)

    def transform(self, data):
        return self._estimate_data_to_cluster_affinity(self._normalize_data(data))
