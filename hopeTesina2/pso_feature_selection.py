import numpy as np
import matplotlib.pyplot as plt
import time
from particle import Particle  
from scipy.stats import spearmanr

def fitness_function(features_selected, data):
    """
    Funzione fitness basata sulla correlazione di Spearman tra le feature selezionate.
    """
    if np.sum(features_selected) == 0:
        return 0  # Evita subset vuoti
    selected_data = data[:, features_selected == 1]
    correlation, _ = spearmanr(selected_data, axis=0)
    correlation = np.abs(correlation)
    return np.mean(correlation[np.triu_indices_from(correlation, k=1)])

class PSOFeatureSelection:
    def __init__(self, num_particles, num_features, data, subset_size=30, max_iter=100, 
                 w=0.7, c1=2.0, c2=2.0, early_stop=False , threshold=10, toll=0, seed=42):
        self.seed = seed
        np.random.seed(self.seed)
        self.num_particles = num_particles
        self.num_features = num_features
        self.subset_size = subset_size
        self.data = data
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self. early_stop = early_stop
        self.threshold = threshold
        self.toll = toll
        self.swarm = [Particle(num_features, subset_size, seed) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_score = float('-inf')
        self.history_best = []
        self.history_avg = []
        self.feature_selection_count = np.zeros(num_features, dtype=int)
        self.hist_velocity = []
        

    def optimize(self):

        
        start_time = time.time()

        convergence_iterator = 0

        # Apri il file di log
        with open("pso_log.txt", "w") as log_file:
            for iter in range(self.max_iter):
                iter_start_time = time.time()
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

                    sigmoid = 1 / (1 + np.exp(-particle.velocity))
                    probabilities = np.random.rand(self.num_features)
                    new_position = np.zeros(self.num_features, dtype=int)
                    selected_indices = np.argsort(sigmoid)[-self.subset_size:]
                    new_position[selected_indices] = 1
                    particle.position = new_position

                    # Update feature selection count
                    self.feature_selection_count += particle.position
                
                self.hist_velocity.append(np.mean(hist_vel_tmp))
                avg_score = np.mean(scores)
                self.history_best.append(self.global_best_score)
                self.history_avg.append(avg_score)

                
                # Scrivi i messaggi di debug nel file di log
                iter_end_time = time.time()
                iter_duration = iter_end_time - iter_start_time
                log_file.write(f"Iteration {iter + 1}/{self.max_iter}\n")
                log_file.write(f"Global Best Score: {self.global_best_score}\n")
                log_file.write(f"Global Best Position: {self.global_best_position}\n")
                log_file.write(f"Average Score: {avg_score}\n")
                log_file.write(f"Iteration Duration: {iter_duration:.2f} seconds\n")
                log_file.write("-" * 50 + "\n")


                # Early stopping condition
                if self.early_stop:
                    if len(self.history_avg) > 1 and abs(self.history_avg[-1] - self.history_avg[-2]) < self.toll:
                        convergence_iterator += 1
                    else:
                        convergence_iterator = 0

                    if convergence_iterator >= self.threshold:
                        log_file.write("Early stopping condition reached.\n")
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
            "global_best_score": self.global_best_score
        }

        # Scrivi la durata totale nel file di log
        with open("pso_log.txt", "a") as log_file:
            log_file.write(f"Total Optimization Duration: {total_duration:.2f} seconds\n")


        return params


        