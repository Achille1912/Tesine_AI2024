import numpy as np
import random

class Particle:
    """
    Represents a particle in a swarm optimization process.

    A particle is a basic entity used in swarm optimization algorithms. Each particle
    has a position, velocity, and remembers its best position found so far. The primary
    purpose of the Particle class is to initialize and maintain these attributes to
    simulate the movement and exploration of the particle in a multidimensional
    solution space.

    :ivar position: Current position of the particle in the feature space represented
        as a numpy array where each element indicates the selection status of a feature.
    :type position: numpy.ndarray
    :ivar velocity: Current velocity of the particle in the solution space represented
        as a numpy array of floating-point values, indicating the direction and magnitude
        of change for each feature.
    :type velocity: numpy.ndarray
    :ivar best_position: Best position found by the particle so far, stored as a
        numpy array. This helps guide the particle towards promising regions of the
        solution space.
    :type best_position: numpy.ndarray
    :ivar best_score: The best score achieved by the particle based on its best
        position. This value is used to track the performance of the particle.
    :type best_score: float
    :ivar num_features: Total number of features in the solution space. Defines the
        dimensionality of the problem the particle explores.
    :type num_features: int
    :ivar particle_size: Number of features selected by the particle in its initial
        position. Determines the subset size of features under consideration.
    :type particle_size: int
    """
    def __init__(self, num_features, particle_size=50, seed=42):
        np.random.seed(seed)
        self.num_features = num_features
        self.particle_size = particle_size
        self.position = np.zeros(num_features, dtype=int)
        selected_indices = np.random.choice(num_features, particle_size, replace=False)
        self.position[selected_indices] = 1  
        self.velocity = np.random.uniform(-1, 1, num_features)
        self.best_position = np.copy(self.position)
        self.best_score = float('-inf')
