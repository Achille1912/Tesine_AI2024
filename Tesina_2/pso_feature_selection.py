import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from particle import Particle  
from scipy.stats import spearmanr
from utils import normalized_exploitation, normalized_exploration 


def fitness_function(features_selected, data):
    """
    Evaluates the "fitness" of a given subset of features based on their Spearman correlation.
    The higher the average absolute correlation between the features, then the higher
    the fitness value. If no features are selected, the function will return 0
    to avoid computation on empty subsets.

    :param features_selected: Binary array or list denoting selected features,
        where 1 indicates a feature is selected and 0 indicates it is not.
    :param data: Numpy array of shape (n_samples, n_features) representing the dataset.
        Rows correspond to samples, and columns correspond to features.
    :return: The average absolute Spearman correlation of the selected features. If
        no features are selected, returns 0.
    """
    if np.sum(features_selected) == 0:
        return 0  # Avoid empty subsets
    selected_data = data[:, features_selected == 1]
    correlation, _ = spearmanr(selected_data, axis=0)
    correlation = np.abs(correlation)
    return np.mean(correlation[np.triu_indices_from(correlation, k=1)])

class PSOFeatureSelection:
    def __init__(self, swarm_size, num_features, data, particle_size, max_iter=100, 
                 w=0.7, c1=2.0, c2=2.0, early_stop=False , threshold=10, toll=0, seed=42):
        self.seed = seed
        np.random.seed(self.seed)
        self.particle_size = particle_size
        self.num_features = num_features
        self.swarm_size = swarm_size
        self.data = data
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.threshold = threshold
        self.toll = toll
        self.swarm = [Particle(num_features, particle_size, seed) for _ in range(swarm_size)]
        self.global_best_position = None
        self.global_best_score = float('-inf')
        self.history_best = []
        self.history_avg = []
        self.history_std = []
        self.feature_selection_count = np.zeros(self.num_features, dtype=int)
        self.hist_velocity = []
        self.hist_exploration = []
        self.hist_exploitation = []
        

    def optimize(self):

        
        start_time = time.time()

        convergence_iterator = 0


        
        for iter in range(self.max_iter):
            scores = []
            hist_vel_tmp = []
            for particle in self.swarm:
                fitness = fitness_function(particle.position, self.data)
                scores.append(fitness)
                
                if fitness > particle.best_score:
                    particle.best_score = fitness
                    particle.best_position = np.copy(particle.position)
                
                if fitness > self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = np.copy(particle.position)
                
                r1, r2 = np.random.rand(self.num_features), np.random.rand(self.num_features)
                cognitive_component = self.c1 * r1 * (particle.best_position - particle.position)
                social_component = self.c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_component + social_component
                
                hist_vel_tmp.append(particle.velocity)

                # Binarize the particle's velocity to determine which features to select:
                # 1. Compute the logistic sigmoid for each velocity component, mapping real values to (0,1).
                sigmoid = 1 / (1 + np.exp(-particle.velocity))
                # 2. Initialize a new position vector of zeros (no features selected).
                new_position = np.zeros(self.num_features, dtype=int)
                # 3. Sort the sigmoid values and take the top `particle_size` indices (the highest probabilities).
                selected_indices = np.argsort(sigmoid)[-self.particle_size:]
                # 4. Set those indices to 1 (i.e., select those features).
                new_position[selected_indices] = 1
                # 5. Update the particle's position with the new binary vector.
                particle.position = new_position

                # Increase the global count of how many times each feature is selected.
                # Here, 'particle.position' is a binary vector indicating which features
                # this particle selected during the current iteration. 
                # By adding it, we accumulate the frequency with which 
                # each feature is chosenacross all particles and iterations.
                self.feature_selection_count += particle.position

            # Calculate the standard deviation of the positions
            self.history_std.append(np.mean(np.std([particle.position for particle in self.swarm], axis=0)))
            
            self.hist_velocity.append(np.mean(hist_vel_tmp))
            avg_score = np.mean(scores)
            self.history_best.append(self.global_best_score)
            self.history_avg.append(avg_score)
            self.hist_exploration.append(normalized_exploration(self.swarm))
            self.hist_exploitation.append(normalized_exploitation(self.swarm, self.global_best_position))
            

            
            # Early stopping condition
            if self.early_stop:
                if len(self.history_avg) > 1 and abs(self.history_avg[-1] - self.history_avg[-2]) < self.toll:
                    convergence_iterator += 1
                else:
                    convergence_iterator = 0

                if convergence_iterator >= self.threshold:
                    print(f"Early stopping condition reached at {iter}.\n")
                    break

        
        total_duration = time.time() - start_time

        
        ##############################################
        params = {
            "history_best": self.history_best,
            "history_avg": self.history_avg,
            "hist_velocity": self.hist_velocity,
            "feature_selection_count": self.feature_selection_count,
            "total_duration": total_duration,
            "global_best_position": self.global_best_position,
            "global_best_score": self.global_best_score,
            "total_duration": total_duration,
            "hist_std": self.history_std,
            "hist_exploration" : self.hist_exploration,
            "hist_exploitation" : self.hist_exploitation
        }



        return params


