import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
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
    RUN = 30
    SEED = 42
    random.seed(42)

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
        
        best_params_dict = None

        for run in range(RUN):
            pso = PSOFeatureSelection(num_particles, num_features, data, subset_size=user_swarm_size, max_iter=user_iterations, 
                                w=user_w, c1=user_c1, c2=user_c2, early_stop=user_early_stop,
                                threshold=user_threshold, toll=user_toll, seed=(SEED+run))            
            params_dict = pso.optimize()
            params_dict["run"] = run+1
            print(f"Run {run+1}/{RUN} completed.")
            
            if best_params_dict is None or params_dict["global_best_score"] > best_params_dict["global_best_score"]:
                best_params_dict = params_dict
        
        selected_feature_names = df.columns[best_params_dict["global_best_position"] == 1]
        print("Best feature subset:", selected_feature_names)
        print("Best fitness score:", best_params_dict["global_best_score"])


        # Create a single figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"PSO with swarm_size={user_swarm_size}, w={user_w}, c1={user_c1}, c2={user_c2}")

        # Plot the history of best and average fitness scores
        axs[0, 0].plot(best_params_dict["history_best"], label="Best Solution", color='b')
        axs[0, 0].plot(best_params_dict["history_avg"], label="Avg Solution", color='r')
        axs[0, 0].set_xlabel("Iterations")
        axs[0, 0].set_ylabel("Fitness Score")
        axs[0, 0].set_title("Fitness Score Over Iterations")
        axs[0, 0].legend()

        # Boxplot for fitness score distribution
        axs[0, 1].boxplot(best_params_dict["history_avg"])
        axs[0, 1].set_title("Distribution of Fitness Scores")
        axs[0, 1].set_ylabel("Fitness Score")

        # Plot average velocity over iterations
        axs[1, 0].plot(best_params_dict["hist_velocity"])
        axs[1, 0].set_xlabel("Iterations")
        axs[1, 0].set_ylabel("Average Velocity")
        axs[1, 0].set_title("Average Velocity Over Iterations")

        # Plot feature selection frequency
        axs[1, 1].bar(range(num_features), best_params_dict["feature_selection_count"])
        axs[1, 1].set_xlabel("Feature Index")
        axs[1, 1].set_ylabel("Selection Frequency")
        axs[1, 1].set_title("Feature Selection Frequency")

        plt.tight_layout()
        plt.show()


        # Ask the user if they want to run another test
        repeat = input("Do you want to run another test? (y/n): ").lower()
        if repeat != 'y':
            print("🔚 End of execution.")
            break












