import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time

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
    def __init__(self, num_particles, num_features, data, subset_size=30, w=0.7, c1=2.0, c2=2.0, max_iter=100):
        self.num_particles = num_particles
        self.num_features = num_features
        self.subset_size = subset_size
        self.data = data
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
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
        
        total_duration = time.time() - start_time
        plt.ioff()
        plt.show()

        # Boxplot for fitness score distribution
        plt.figure()
        plt.boxplot(scores)
        plt.title("Distribution of Fitness Scores")
        plt.ylabel("Fitness Score")
        plt.show()

        print(f"Total Optimization Duration: {total_duration:.2f} seconds")

        plt.figure()
        plt.plot(self.hist_velocity)
        plt.xlabel("Iterations")
        plt.ylabel("Average Velocity")
        plt.title("Average Velocity Over Iterations")
        plt.show()


        
        return self.global_best_position, self.global_best_score

# Esempio di utilizzo
if __name__ == "__main__":
    num_particles = 50
    max_iter = 100

    # Caricamento del dataset
    df = pd.read_csv("DARWIN.csv")

    # Selezione solo delle colonne numeriche, escludendo la prima e l'ultima colonna
    df = df.select_dtypes(include=[np.number]).iloc[:, 1:-1]

    # Gestione valori mancanti: rimpiazziamo con la media della colonna
    df.fillna(df.mean(), inplace=True)

    # Normalizzazione (opzionale)
    df = (df - df.min()) / (df.max() - df.min())

    # Conversione in numpy array per PSO
    data = df.to_numpy()

    # Aggiorniamo num_features per il numero corretto di feature disponibili
    num_features = data.shape[1]
    subset_size = 50  # Numero di feature da selezionare per ogni particella
    
    pso = PSOFeatureSelection(num_particles, num_features, data, subset_size=subset_size, max_iter=max_iter)
    best_features, best_score = pso.optimize()
    
    selected_feature_names = df.columns[best_features == 1]
    print("Miglior subset di feature:", selected_feature_names)
    print("Punteggio fitness:", best_score)
