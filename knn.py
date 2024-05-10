import numpy as np
import pandas as pd

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _get_neighbors(self, x):
        distances = [(self._euclidean_distance(x, x_train), y_train) for x_train, y_train in zip(self.X_train, self.y_train)]
        distances.sort(key=lambda x: x[0])
        neighbors = [distances[i][1] for i in range(self.n_neighbors)]
        # Add a check to convert string labels to integers
        if isinstance(neighbors[0], str):
            neighbors = [int(neighbor) for neighbor in neighbors]
        return neighbors


    def _predict_instance(self, x):
        neighbors = self._get_neighbors(x)
        counts = np.bincount(neighbors)
        return np.argmax(counts)

    def predict(self, X):
        # Convert one-hot encoded data back to numpy array
        X = X.values if isinstance(X, pd.DataFrame) else X
        return [self._predict_instance(x) for x in X]