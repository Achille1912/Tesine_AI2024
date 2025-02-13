import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Verifica se il file esiste
file_path = 'src/DARWIN.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Il file {file_path} non è stato trovato. Assicurati di caricarlo nella posizione corretta.")

# Caricamento del dataset
df = pd.read_csv(file_path)

# Selezioniamo solo le colonne numeriche
num_features = df.select_dtypes(include=[np.number]).columns.tolist()



# Funzione di fitness: minimizza la correlazione media tra le features selezionate
def fitness_function(selected_features):
    selected_features = [num_features[int(i)] for i in selected_features]
    selected_data = df[selected_features]
    corr_matrix = selected_data.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    mean_corr = upper_triangle.stack().mean()
    return mean_corr  # Minimizziamo questa quantità

def decode_binary_chromosome(binary_string, num_features):
    """
    Decodifica una stringa binaria in un elenco di feature selezionate.
    
    Args:
    binary_string (str): La stringa binaria che rappresenta le feature selezionate.
    num_features (list): L'elenco delle feature disponibili.
    
    Returns:
    list: L'elenco delle feature selezionate.
    """
    # Binary string è una matrice binaria con 30 colonne e 450 righe
    # Voglio mettere l'indice di riga ad ogni "1"

    # Ottenendo però in uscita una matrice con 30 colonne e 50 righe
    # perchè devo selezionare solo 50 features

    selected_features = []
    

    for particle in binary_string:
        index_particle = []
        c = 0
        for i in range(len(particle)):
            if particle[i] == 1:
                index_particle.append(i)
                c +=1
        print(c)
        selected_features.append(index_particle)
        

                
    return selected_features

def encode_binary_chromosome(selected_features, num_features):
    """
    Codifica un elenco di feature selezionate in una lista binaria.
    
    Args:
    selected_features (list): L'elenco delle feature selezionate.
    num_features (list): L'elenco delle feature disponibili.
    
    Returns:
    list: La lista binaria che rappresenta le feature selezionate.
    """
    binary_list = [0] * len(num_features)
    for index in selected_features:
        binary_list[int(index)] = 1
    return binary_list


def fitness_wrapper(chromosome: np.ndarray) -> float:
    """Wrapper for fitness function to handle binary chromosome."""
    params = decode_binary_chromosome(chromosome, num_features)
    return fitness_function(params)


def particle_swarm_optimization(num_particles, max_iter, w, c1, c2, dim, lb, ub, fitness_function, initial_positions, initial_velocities):
    # Inizializza posizioni e velocità dei particles
    #positions = decode_binary_chromosome(initial_positions, num_features)
    positions = initial_positions
    velocities = initial_velocities

    # Inizializza il miglior risultato personale per ciascun particle
    pbest_positions = positions.copy()
    pbest_scores = np.array([fitness_function(p) for p in positions]) 

    # Trova il miglior risultato globale iniziale
    gbest_index = np.argmin(pbest_scores) 
    gbest_score = pbest_scores[gbest_index] 
    gbest_position = pbest_positions[gbest_index] 

    # Prepara la visualizzazione interattiva con due linee: best e average
    plt.ion()
    fig, ax = plt.subplots()
    line_best, = ax.plot([], [], 'bo-', label="Global Best Fitness")
    line_avg, = ax.plot([], [], 'ro-', label="Average Fitness")
    ax.set_xlabel("Iterazione")
    ax.set_ylabel("Fitness (inverted)")
    ax.set_title("Convergenza PSO")
    ax.grid(True)
    ax.legend()

    gbest_history = []
    avg_history = []

    # Ciclo PSO con aggiornamento grafico ad ogni iterazione
    for iteration in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            # Aggiornamento velocità
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))
            # Aggiornamento posizione
            positions[i] += velocities[i]
            # Assicura che le posizioni restino nei limiti
            positions[i] = np.clip(positions[i], lb, ub)
            
            score = fitness_function(positions[i])
            # Aggiorna il miglior personale se il nuovo score è migliore
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i]
                # Aggiorna il miglior globale se necessario
                if score < gbest_score:
                    gbest_score = score
                    gbest_position = positions[i]
        
        # Calcola l'average fitness a partire dalle fitness personali
        avg_fitness = np.mean(pbest_scores)
        gbest_history.append(1 / gbest_score)  # Inverti il valore di fitness
        avg_history.append(1 / avg_fitness)  # Inverti il valore di fitness
        
        # Aggiornamento della trama
        line_best.set_data(range(len(gbest_history)), gbest_history)
        line_avg.set_data(range(len(avg_history)), avg_history)
        ax.set_xlim(0, len(gbest_history) + 1)
        ax.set_ylim(0, max(max(gbest_history), max(avg_history)) * 1.05)
        plt.draw()
        plt.pause(0.1)

        # Stampa le particelle scelte ad ogni iterazione
        print(f"Iterazione {iteration + 1}:")
        for j in range(num_particles):
            selected_features = encode_binary_chromosome(positions[j], num_features)
            print(f"  Particella {j + 1}: {selected_features}")


    plt.ioff()
    plt.show()

    # Features selezionate
    selected_features = [num_features[int(i)] for i in gbest_position]
    print("Ottimizzazione completata! Features selezionate:", selected_features)
    return selected_features

# Parametri PSO
num_particles = 30
max_iter = 100
w = 0.7   # peso d'inerzia
c1 = 2  # coefficiente cognitivo
c2 = 2  # coefficiente sociale

# Dimensione dello spazio delle particelle
dim = 50  # 50 features da selezionare
lb = [0] * dim  # Limite inferiore (prima feature)
ub = [len(num_features) - 1] * dim  # Limite superiore (ultima feature)

initial_positions = np.random.uniform(low=lb, high=ub, size=(num_particles, dim))
#initial_positions = np.random.choice([0, 1], size=(num_particles, len(num_features)))
initial_velocities = np.random.uniform(low=-1, high=1, size=(num_particles, dim))



# Esegui l'ottimizzazione PSO
selected_features = particle_swarm_optimization(num_particles, 
                                                max_iter, w, c1, c2, dim, lb, ub, 
                                                fitness_function, initial_positions, 
                                                initial_velocities)

