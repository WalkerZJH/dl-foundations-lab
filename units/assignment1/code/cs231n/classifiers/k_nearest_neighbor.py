from builtins import object
from builtins import range

import numpy as np


class KNearestNeighbor(object):
    """k-nearest neighbor classifier with L2 distance."""

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """Memorize the training data for nearest-neighbor lookup."""
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """Predict labels for test data."""
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """Compute L2 distances using nested loops."""
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                diff = X[i] - self.X_train[j]
                dists[i, j] = np.sqrt(np.sum(diff * diff))
        return dists

    def compute_distances_one_loop(self, X):
        """Compute L2 distances using one loop over test examples."""
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            diff = self.X_train - X[i]
            dists[i, :] = np.sqrt(np.sum(diff * diff, axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        """Compute L2 distances without explicit Python loops."""
        test_sq = np.sum(X * X, axis=1, keepdims=True)
        train_sq = np.sum(self.X_train * self.X_train, axis=1)
        cross = X.dot(self.X_train.T)
        return np.sqrt(np.maximum(test_sq + train_sq - 2 * cross, 0.0))

    def predict_labels(self, dists, k=1):
        """Predict labels from a distance matrix."""
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=int)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            counts = np.bincount(closest_y.astype(int))
            y_pred[i] = np.argmax(counts)
        return y_pred
