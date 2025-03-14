# PSO Feature Selection Project Documentation

## Introduzione

Questo progetto utilizza l'ottimizzazione dello sciame di particelle (PSO) per la selezione delle feature in un dataset. L'obiettivo è selezionare un sottoinsieme di feature che massimizza la correlazione di Spearman tra le feature selezionate. Il progetto è composto da diversi file, ciascuno con un ruolo specifico.

## Struttura del Progetto

- `main.py`: Il file principale che esegue il PSO e gestisce i parametri dell'utente.
- `pso_feature_selection.py`: Contiene la classe PSOFeatureSelection che implementa l'algoritmo PSO.
- `utils.py`: Contiene funzioni di utilità come la funzione di fitness e le metriche di esplorazione ed esploitazione.
- `plot_utils.py`: Contiene la funzione per la visualizzazione dei risultati.
- `Exp.ipynb`: Un notebook Jupyter per l'analisi dei risultati.

## main.py

### Descrizione

Il file `main.py` è il punto di ingresso del progetto. Qui vengono definiti i parametri disponibili per l'utente e viene eseguito il PSO. I risultati migliori vengono salvati e visualizzati.

### Funzionamento

1. **Definizione dei Parametri**: Vengono definiti i parametri disponibili per l'utente, come la dimensione dello sciame, i coefficienti di accelerazione, il numero massimo di iterazioni, ecc.
2. **Caricamento del Dataset**: Il dataset viene caricato e preprocessato, selezionando solo le colonne numeriche e gestendo i valori mancanti.
3. **Esecuzione del PSO**: Viene eseguito il PSO per un numero specificato di run. I risultati di ogni run vengono salvati e il miglior risultato viene selezionato.
4. **Visualizzazione dei Risultati**: I risultati vengono visualizzati utilizzando la funzione `plot_results` dal file `plot_utils.py`.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import psutil
from pso_feature_selection import PSOFeatureSelection
from scenarios import params_selection
from plot_utils import plot_results  # Import the plotting function

if __name__ == "__main__":

    # Definition of available parameters
    swarm_sizes = [20, 50, 100, 200, 500]
    w_values = [0.4, 0.6, 0.7, 0.9]
    c1_values = [1.0, 1.5, 2.0, 2.5]
    c2_values = [1.0, 1.5, 2.0, 2.5]
    iterations_values = [50,100,200]
    bool_early_stop = [True, False]
    threshold_values = [10,20,30]
    toll_values = [0.0001, 0.00001, 0.000001]

    # Particle size and other parameters
    num_particles = 50
    iterations = 100
    RUN = 2
    SEED = 42
    random.seed(42)

    # Load the dataset
    df = pd.read_csv("DARWIN.csv")

    # Select only numeric columns, excluding the first and last columns
    df = df.select_dtypes(include=[np.number]).iloc[:, 1:-1]

    # Handle missing values: replace with column mean
    df.fillna(df.mean(), inplace=True)

    # Normalization (optional)
    df = (df - df.min()) / (df.max() - df.min())

    # Convert to numpy array for PSO
    data = df.to_numpy()

    # Update num_features to the correct number of available features
    num_features = data.shape[1]

    while True:
        print("\n=== PSO PARAMETER CONFIGURATION ===")
        user_threshold = 0
        user_toll = 0

        user_swarm_size = params_selection("user_swarm_size", swarm_sizes)
        if user_swarm_size is None: break

        user_w = params_selection("w", w_values)
        if user_w is None: break

        user_c1 = params_selection("user_c1", c1_values)
        if user_c1 is None: break

        user_c2 = params_selection("user_c2", c2_values)
        if user_c2 is None: break

        user_iterations = params_selection("max_iter", iterations_values)
        if user_iterations is None: break

        user_early_stop = params_selection("early_stop", bool_early_stop)
        if user_early_stop is None: break

        if user_early_stop:
            user_threshold = params_selection("threshold", threshold_values)
            if user_threshold is None: break

            user_toll = params_selection("tollerance", toll_values)
            if user_toll is None: break

        print(f"\n🚀 Running PSO with user_swarm_size={user_swarm_size}, w={user_w}, user_c1={user_c1}, user_c2={user_c2}\n")

        best_params_dict = None

        run_dict = []

        for run in range(RUN):
            pso = PSOFeatureSelection(num_particles, num_features, data, subset_size=user_swarm_size, max_iter=user_iterations,
                                w=user_w, c1=user_c1, c2=user_c2, early_stop=user_early_stop,
                                threshold=user_threshold, toll=user_toll, seed=(SEED+run))
            params_dict = pso.optimize()
            params_dict["run"] = run+1
            run_dict.append(params_dict)

            if best_params_dict is None or params_dict["global_best_score"] > best_params_dict["global_best_score"]:
                best_params_dict = params_dict
            print(f"Run {run+1}/{RUN} completed.")

        # Calculate stability between runs
        stability_scores = []
        for i in range(1, len(run_dict)):
            stability = np.mean(run_dict[i]["history_avg"]) / np.mean(run_dict[i-1]["history_avg"])
            stability_scores.append(stability)
            print(f"Stability between run {i} and run {i+1}: {stability:.4f}")

        avg_stability = np.mean(stability_scores)
        print(f"Average stability across runs: {avg_stability:.4f}")


        selected_feature_names = df.columns[best_params_dict["global_best_position"] == 1]
        print("Best feature subset:", selected_feature_names)
        print("Best fitness score:", best_params_dict["global_best_score"])

        process = psutil.Process(os.getpid())
        memory_used = round(process.memory_info().rss / 1024 ** 2, 2)
        print(f"Memory used: {memory_used} MB")
        total_duration = best_params_dict["total_duration"]

        # Call the plotting function
        plot_results(best_params_dict, run_dict, stability_scores, df, user_swarm_size, user_w, user_c1, user_c2, memory_used, total_duration)

        # Ask the user if they want to run another test
        repeat = input("Do you want to run another test? (y/n): ").lower()
        if repeat != 'y':
            print("🔚 End of execution.")
            break
```

## pso_feature_selection.py

### Descrizione

Il file `pso_feature_selection.py` contiene la classe `PSOFeatureSelection` che implementa l'algoritmo PSO per la selezione delle feature.

### Funzionamento

1. **Inizializzazione**: Vengono inizializzati i parametri del PSO, come la dimensione dello sciame, il numero di feature, i coefficienti di accelerazione, ecc.
2. **Ottimizzazione**: La funzione `optimize` esegue l'algoritmo PSO. Ad ogni iterazione, le particelle aggiornano le loro posizioni e velocità in base alle componenti cognitive e sociali. Viene calcolata la fitness di ogni particella e vengono aggiornate le posizioni migliori locali e globali.
3. **Condizione di Arresto Anticipato**: Se abilitata, la condizione di arresto anticipato interrompe l'ottimizzazione se la variazione della fitness media tra le iterazioni è inferiore a una soglia specificata.
4. **Salvataggio dei Risultati**: I risultati dell'ottimizzazione vengono salvati in un dizionario e restituiti.

## utils.py

### Descrizione

Il file `utils.py` contiene funzioni di utilità utilizzate nel progetto.

### Funzioni

1. **hamming_distance**: Calcola la distanza di Hamming tra due vettori.
2. **normalized_exploitation**: Calcola l'esploitazione normalizzata come la distanza media di Hamming tra le particelle e la migliore soluzione globale.
3. **normalized_exploration**: Calcola l'esplorazione normalizzata come la distanza media di Hamming tra tutte le coppie di particelle.
4. **fitness_function**: Calcola la fitness basata sulla correlazione di Spearman tra le feature selezionate.

## plot_utils.py

### Descrizione

Il file `plot_utils.py` contiene la funzione `plot_results` per la visualizzazione dei risultati.

### Funzionamento

1. **Creazione dei Grafici**: Vengono creati diversi grafici per visualizzare i risultati dell'ottimizzazione, come la storia della fitness migliore e media, la velocità media delle particelle, la frequenza di selezione delle feature, ecc.
2. **Salvataggio dei Grafici**: I grafici vengono salvati come file PNG nella cartella `plots`.

## Exp.ipynb

### Descrizione

Il file `Exp.ipynb` è un notebook Jupyter utilizzato per l'analisi dei risultati.

### Funzionamento

1. **Lettura dei Dati**: I dati vengono letti dal file `pso_log.txt`.
2. **Parsing dei Dati**: I dati vengono parsati in una lista di liste.
3. **Visualizzazione dei Grafici**: Vengono creati grafici per visualizzare i risultati dell'ottimizzazione.

## Conclusione

Questo progetto utilizza l'algoritmo PSO per la selezione delle feature in un dataset. L'implementazione include la gestione dei parametri dell'utente, l'esecuzione del PSO, la visualizzazione dei risultati e l'analisi dei risultati. Ogni file ha un ruolo specifico e contribuisce al funzionamento complessivo del progetto.
