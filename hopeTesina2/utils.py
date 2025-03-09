import numpy as np


def hamming_distance(x, y):
    return np.sum(np.abs(x - y))

def normalized_exploitation(population, best_solution):
    dim = len(best_solution)
    mean_hamming = np.mean([hamming_distance(ind.position, best_solution) for ind in population])
    return 1 - (mean_hamming / dim)

def normalized_exploration(population):
    num_particles = len(population)
    dim = len(population[0].position)
    mean_pairwise_distance = np.mean([
        hamming_distance(population[i].position, population[j].position)
        for i in range(num_particles) for j in range(i + 1, num_particles)
    ])
    return mean_pairwise_distance / dim  # Normalize between 0 and 1

