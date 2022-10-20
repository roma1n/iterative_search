import numpy as np
from scipy import stats as sps

from lib.ml.dist.dist import Dist


class ZeroInflattedPoisson(Dist):
    '''https://en.wikipedia.org/wiki/Zero-inflated_model'''
    def __init__(self):
        super().__init__()

    def fit(self, data, weights=None):
        average, std = self._calculate_moments(data, weights)
        pi_devisor = average ** 2 + std ** 2 - average
        self.params = {
            'lambda': max(0., (average ** 2 + std ** 2) / average - 1) if average != 0 else 0.,
            'pi': max(0., min(1., (std ** 2 - average) / pi_devisor)) if pi_devisor != 0 else 1.,
        }

    def random_fit(self, data):
        random_sample = None
        while random_sample is None or len(np.unique(random_sample)) < 2:
            random_sample = np.random.choice(data, size=3, replace=True)
        self.fit(random_sample)

    def proba(self, data):
        data = np.array(data)
        return self.params['pi'] * (data == 0).astype(float) + (1 - self.params['pi']) \
            * sps.poisson(self.params['lambda']).pmf(data)
