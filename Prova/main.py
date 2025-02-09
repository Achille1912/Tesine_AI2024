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

# Assicuriamoci di avere almeno 50 features
if len(num_features) < 50:
    raise ValueError("Il dataset ha meno di 50 features numeriche, impossibile procedere.")

# Funzione di fitness: minimizza la correlazione media tra le features selezionate
def fitness_function(selected_features):
    selected_features = [num_features[int(i)] for i in selected_features]
    selected_data = df[selected_features]
    corr_matrix = selected_data.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    mean_corr = upper_triangle.stack().mean()
    return mean_corr  # Minimizziamo questa quantità

# Dimensione dello spazio delle particelle
dim = 50  # 50 features da selezionare
lb = [0] * dim  # Limite inferiore (prima feature)
ub = [len(num_features) - 1] * dim  # Limite superiore (ultima feature)

# Parametri PSO
num_particles = 30
max_iter = 100
w = 0.7   # peso d'inerzia
c1 = 2  # coefficiente cognitivo
c2 = 2  # coefficiente sociale

# Inizializza posizioni e velocità dei particles
positions = np.random.uniform(low=lb, high=ub, size=(num_particles, dim))
velocities = np.random.uniform(low=-1, high=1, size=(num_particles, dim))

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
ax.set_ylabel("Fitness")
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
    gbest_history.append(gbest_score)
    avg_history.append(avg_fitness)
    
    # Aggiornamento della trama
    line_best.set_data(range(len(gbest_history)), gbest_history)
    line_avg.set_data(range(len(avg_history)), avg_history)
    ax.set_xlim(0, len(gbest_history) + 1)
    ax.set_ylim(0, max(max(gbest_history), max(avg_history)) * 1.05)
    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()

# Features selezionate
selected_features = [num_features[int(i)] for i in gbest_position]
print("Ottimizzazione completata! Features selezionate:", selected_features)
