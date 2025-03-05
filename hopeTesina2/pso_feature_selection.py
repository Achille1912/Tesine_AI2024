import numpy as np
import matplotlib.pyplot as plt
import time
from particle import Particle  

def fitness_function(features_selected, data):
    """
    Funzione fitness basata sulla correlazione tra le feature selezionate.
    """
    if np.sum(features_selected) == 0:
        return 0  # Evita subset vuoti
    selected_data = data[:, features_selected == 1]
    correlation = np.abs(np.corrcoef(selected_data, rowvar=False))
    return np.mean(correlation[np.triu_indices_from(correlation, k=1)])

class PSOFeatureSelection:
    def __init__(self, num_particles, num_features, data, subset_size=30, max_iter=100, 
                 w=0.7, c1=2.0, c2=2.0, early_stop=False , threshold=10, toll=0):
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
        self.swarm = [Particle(num_features, subset_size) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_score = float('-inf')
        self.history_best = []
        self.history_avg = []
        self.feature_selection_count = np.zeros(num_features, dtype=int)
        self.hist_velocity = []
        

    def optimize(self):
        plt.ion()
        fig, ax = plt.subplots()
        
        start_time = time.time()

        convergence_iterator = 0
        
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

            
            
            ax.clear()
            ax.plot(self.history_best, label="Best Solution", color='b')
            ax.plot(self.history_avg, label="Avg Solution", color='r')
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Fitness Score")
            ax.legend()
            plt.pause(0.1)
            
            # Print statements for debugging
            iter_end_time = time.time()
            iter_duration = iter_end_time - iter_start_time
            print(f"Iteration {iter + 1}/{self.max_iter}")
            print(f"Global Best Score: {self.global_best_score}")
            print(f"Global Best Position: {self.global_best_position}")
            print(f"Average Score: {avg_score}")
            print(f"Iteration Duration: {iter_duration:.2f} seconds")
            print("-" * 50)


            # Early stopping condition
            if self.early_stop:
                if len(self.history_avg) > 1 and abs(self.history_avg[-1] - self.history_avg[-2]) < self.toll:
                    convergence_iterator += 1
                else:
                    convergence_iterator = 0

                if convergence_iterator >= self.threshold:
                    print("Early stopping condition reached.")
                    break

        
        total_duration = time.time() - start_time
        plt.ioff()
        plt.show()

        # Create a single figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"PSO with swarm_size={self.subset_size}, w={self.w}, c1={self.c1}, c2={self.c2}")


        # Plot the history of best and average fitness scores
        axs[0, 0].plot(self.history_best, label="Best Solution", color='b')
        axs[0, 0].plot(self.history_avg, label="Avg Solution", color='r')
        axs[0, 0].set_xlabel("Iterations")
        axs[0, 0].set_ylabel("Fitness Score")
        axs[0, 0].set_title("Fitness Score Over Iterations")
        axs[0, 0].legend()

        # Boxplot for fitness score distribution
        axs[0, 1].boxplot(self.history_avg)
        axs[0, 1].set_title("Distribution of Fitness Scores")
        axs[0, 1].set_ylabel("Fitness Score")

        # Plot average velocity over iterations
        axs[1, 0].plot(self.hist_velocity)
        axs[1, 0].set_xlabel("Iterations")
        axs[1, 0].set_ylabel("Average Velocity")
        axs[1, 0].set_title("Average Velocity Over Iterations")

        # Plot feature selection frequency
        axs[1, 1].bar(range(self.num_features), self.feature_selection_count)
        axs[1, 1].set_xlabel("Feature Index")
        axs[1, 1].set_ylabel("Selection Frequency")
        axs[1, 1].set_title("Feature Selection Frequency")

        plt.tight_layout()
        plt.show()

        print(f"Total Optimization Duration: {total_duration:.2f} seconds")

        return self.global_best_position, self.global_best_score