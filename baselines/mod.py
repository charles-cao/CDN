"""Mean-shift outlier detection (MOD) adapted to the semi-supervised setting.

Original method and code: J. Yang, S. Rahardja, P. Franti, Mean-shift outlier
detection and filtering, Pattern Recognition 115 (2021) 107874.

The original MOD shifts every point of one dataset towards the mean of its
k nearest neighbours within that dataset. Here the training set of normal
data is kept fixed as the reference: each query is shifted towards the mean
of its k nearest neighbours in the training set, for a fixed number of
iterations, and scored by the distance between its original and final
positions. The paper uses k = 5 and the original three iterations.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class MOD:
    def __init__(self, k=5, iteration_number=3):
        self.k = k
        self.iteration_number = iteration_number

    def fit(self, X, y=None):
        """Store the normal training set as the reference for shifting."""
        self.data = np.array(X, dtype=float)
        self.nbrs_ = NearestNeighbors(n_neighbors=self.k).fit(self.data)
        return self

    def decision_function(self, X):
        """Anomaly scores of X; larger means more anomalous."""
        points = np.array(X, dtype=float)
        shifted = points.copy()
        for _ in range(self.iteration_number):
            idx = self.nbrs_.kneighbors(shifted, return_distance=False)
            shifted = self.data[idx].mean(axis=1)
        return np.linalg.norm(points - shifted, axis=1)
