import numpy as np


def hamming_distance(x, y):
    return np.sum(np.abs(x - y))

def normalized_exploitation(population, best_solution):
    dim = len(best_solution)
    mean_hamming = np.mean([hamming_distance(ind.position, best_solution) for ind in population])
    return 1 - (mean_hamming / dim)

def normalized_exploration(population):
    """
    Calculates the normalized pairwise exploration of a population by computing the
    mean pairwise Hamming distance between the positions of particles in the
    population, normalized by the dimensionality of the position space.

    The function first calculates the dimensionality of the particle positions and
    then determines the mean Hamming distance between pairs of particles within the
    population. The resulting mean distance is normalized by the dimension to
    produce a value between 0 and 1.

    :param population: A list of particle objects. Each particle object must have
        a "position" attribute which is iterable and represents the position of the
        particle in the solution space.
    :returns: A float value between 0 and 1 representing the normalized exploration
        of the population.
    :rtype: float
    """
    num_particles = len(population)
    dim = len(population[0].position)
    mean_pairwise_distance = np.mean([
        hamming_distance(population[i].position, population[j].position)
        for i in range(num_particles) for j in range(i + 1, num_particles)
    ])
    return mean_pairwise_distance / dim 

