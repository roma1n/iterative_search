import numpy as np


class Dist:
    def __init__(self):
        self.params = None

    @staticmethod
    def _calculate_moments(data, weights=None):
        average = np.average(data, weights=weights)
        return average, np.sqrt(np.average((data - average) ** 2, weights=weights))

    def fit(self, data):
        pass

    def random_fit(self, data):
        pass

    def proba(self, data):
        return np.zeros_like(data)

    def __repr__(self):
        return('Dist: {}; params: {}'.format(self.__class__.__name__, str(self.params or '')))
