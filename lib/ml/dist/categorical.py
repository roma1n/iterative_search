import numpy as np
from scipy import stats as sps

from lib.ml.dist.dist import Dist


class Categorical(Dist):
    def __init__(self):
        super().__init__()

    def fit(self, data, weights=None):
        counts = np.bincount(data, weights)
        self.params = {
            'p': counts / counts.sum(),
        }

    def random_fit(self, data):
        self.params = {
            'p': sps.dirichlet(1 + np.bincount(data)).rvs()[0],
        }

    def proba(self, data):
        return self.params['p'][data]
