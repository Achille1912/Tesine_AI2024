import numpy as np

class Particle:
    def __init__(self, num_features, subset_size=30):
        self.num_features = num_features
        self.subset_size = subset_size
        self.position = np.zeros(num_features, dtype=int)
        selected_indices = np.random.choice(num_features, subset_size, replace=False)
        self.position[selected_indices] = 1  # Selezioniamo esattamente subset_size feature
        self.velocity = np.random.uniform(-1, 1, num_features)
        self.best_position = np.copy(self.position)
        self.best_score = float('-inf')