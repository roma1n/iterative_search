import numpy as np
from scipy import stats as sps

from lib.ml.dist.dist import Dist


class Normal(Dist):
    def __init__(self):
        super().__init__()

    def fit(self, data, weights=None):
        average, std = self._calculate_moments(data, weights)
        self.params = {
            'average': average,
            'std': 0.1 + std,
        }

    def random_fit(self, data):
        random_sample = None
        while random_sample is None or len(np.unique(random_sample)) < 2:
            random_sample = np.random.choice(data, size=3, replace=True)
        self.fit(random_sample)

    def proba(self, data):
        return sps.norm(self.params['average'], self.params['std']).pdf(data)
