import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import psutil
from pso_feature_selection import PSOFeatureSelection
from scenarios import params_selection 
from plot_utils import plot_results  # Import the visualization function

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

    #------------------------------------------

    particle_size = 50
    iterations = 100
    RUN = 30
    SEED = 42
    random.seed(SEED)

    # Loading the dataset
    df = pd.read_csv("DARWIN.csv")

    # Select only numeric columns, excluding the first and last column
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

        print(f"\n🚀 Running PSO with user_swarm_size={user_swarm_size}, particle_size={particle_size}, w={user_w}, user_c1={user_c1}, user_c2={user_c2}\n")
        
        best_params_dict = None

        run_dict = []

        for run in range(RUN):
            pso = PSOFeatureSelection(user_swarm_size, num_features, data, particle_size=particle_size, max_iter=user_iterations, 
                                w=user_w, c1=user_c1, c2=user_c2, early_stop=user_early_stop,
                                threshold=user_threshold, toll=user_toll, seed=(SEED+run))            
            params_dict = pso.optimize()
            params_dict["run"] = run
            params_dict["particle_size"] = particle_size
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


        with open("pso_log.txt", "a") as log_file:
            log_file.write(f"swarm size: {user_swarm_size}: {best_params_dict['history_best']}\n")
            #log_file.write(f"avg w {user_w}: {best_params_dict['history_avg']}\n")

        # Call the visualization function
        plot_results(best_params_dict, run_dict, stability_scores, 
                     df, user_swarm_size, user_w, user_c1, user_c2, memory_used, 
                     total_duration, best_params_dict["run"]+SEED, user_iterations, 
                     user_early_stop, user_threshold, user_toll, particle_size)

        # Ask the user if they want to run another test
        repeat = input("Do you want to run another test? (y/n): ").lower()
        if repeat != 'y':
            print("🔚 End of execution.")
            break
