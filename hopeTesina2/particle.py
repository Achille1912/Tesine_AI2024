import numpy as np
import random

class Particle:
    def __init__(self, num_features, particle_size=50, seed=42):
        np.random.seed(seed)
        self.num_features = num_features
        self.particle_size = particle_size
        self.position = np.zeros(num_features, dtype=int)
        selected_indices = np.random.choice(num_features, particle_size, replace=False)
        self.position[selected_indices] = 1  # Selezioniamo esattamente subset_size feature
        self.velocity = np.random.uniform(-1, 1, num_features)
        self.best_position = np.copy(self.position)
        self.best_score = float('-inf')