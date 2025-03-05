import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pso_feature_selection import PSOFeatureSelection
from scenarios import params_selection 

if __name__ == "__main__":
    # Definizione dei parametri disponibili
    swarm_sizes = [20, 50, 100, 200, 500]
    w_values = [0.4, 0.6, 0.7, 0.9]
    c1_values = [1.0, 1.5, 2.0, 2.5]
    c2_values = [1.0, 1.5, 2.0, 2.5]
    iterations_values = [50,100,200]
    bool_early_stop = [True, False]
    threshold_values = [10,20,30]
    toll_values = [0.0001, 0.00001, 0.000001]

    #------------------------------------------

    num_particles = 50
    iterations = 100

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
        
        pso = PSOFeatureSelection(num_particles, num_features, data, subset_size=user_swarm_size, max_iter=user_iterations, 
                              w=user_w, c1=user_c1, c2=user_c2, early_stop=user_early_stop,
                            threshold=user_threshold, toll=user_toll)
        best_features, best_score = pso.optimize()
        
        selected_feature_names = df.columns[best_features == 1]
        print("Best feature subset:", selected_feature_names)
        print("Fitness score:", best_score)

        # Ask the user if they want to run another test
        repeat = input("Do you want to run another test? (y/n): ").lower()
        if repeat != 'y':
            print("🔚 End of execution.")
            break
    
    










        