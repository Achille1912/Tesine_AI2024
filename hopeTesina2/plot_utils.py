import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(best_params_dict, run_dict, stability_scores, df, user_swarm_size, 
                 user_w, user_c1, user_c2, memory_used, total_duration, seed, user_iterations, 
                 user_early_stop, user_threshold, user_toll, particle_size):
    """
    Plots and saves visual analysis of Particle Swarm Optimization (PSO) performance.
    The function generates two figures with subplots illustrating PSO dynamics, including
    fitness scores, velocity, feature selection frequency, swarm diversity, and stability
    metrics. Additionally, the memory usage and total runtime for the PSO process are 
    displayed.

    :param best_params_dict: Dictionary containing performance metrics and historical PSO 
        data, including fields such as "history_best", "history_avg", "global_best_score", 
        "hist_velocity", "feature_selection_count", "hist_std", "hist_exploration", 
        and "hist_exploitation".
    :param run_dict: Dictionary capturing details about different PSO runs or iterations.
    :param stability_scores: List of stability coefficients between consecutive PSO runs.
    :param df: Input DataFrame whose columns correspond to features being optimized.
    :param user_swarm_size: User-defined swarm size representing the number of particles.
    :param user_w: Inertia weight parameter controlling the influence of particle momentum.
    :param user_c1: Cognitive coefficient representing the influence of personal best 
        position.
    :param user_c2: Social coefficient representing the influence of the group's best 
        position.
    :param memory_used: Float value denoting the memory usage during the process, measured 
        in megabytes (MB).
    :param total_duration: Float value representing the total time consumed for running 
        the PSO, measured in seconds.
    :param seed: Integer value used for initializing random seed, ensuring reproducibility.
    :param user_iterations: Total number of PSO iterations if early stopping is disabled.
    :param user_early_stop: Boolean flag indicating whether early stopping criteria 
        is enabled.
    :param user_threshold: Threshold value used to trigger early stopping in the PSO 
        process.
    :param user_toll: Tolerance level used in conjunction with early stopping criteria.
    :param particle_size: Integer representing the dimensionality of particle positions.

    :return: None
    """
    # Create a single figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    if user_early_stop:
        fig.suptitle(f"PSO with swarm_size={user_swarm_size}, particle_size={particle_size}, w={user_w}, c1={user_c1}, c2={user_c2}, seed = {seed}, threshold={user_threshold}, toll={user_toll}")
    else:
        fig.suptitle(f"PSO with swarm_size={user_swarm_size}, particle_size={particle_size}, w={user_w}, c1={user_c1}, c2={user_c2}, seed = {seed}, max_iteration={user_iterations}")

    # Plot the history of best and average fitness scores
    axs[0, 0].plot(best_params_dict["history_best"], label="Best Solution", color='b')
    axs[0, 0].plot(best_params_dict["history_avg"], label="Avg Solution", color='r')
    axs[0, 0].set_xlabel("Iterations")
    axs[0, 0].set_ylabel("Fitness Score")
    axs[0, 0].set_title("Fitness Score Over Iterations (gbest = {:.4f})".format(best_params_dict["global_best_score"]))
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

    # Plot feature selection frequency using boxplots for the top 15 features
    feature_selection_count = best_params_dict["feature_selection_count"]
    top_features_indices = np.argsort(feature_selection_count)[-15:][::-1]  
    feature_selection_data = [feature_selection_count[feature] for feature in top_features_indices]

    axs[1, 1].bar(range(1, 16), feature_selection_data, color='green')
    axs[1, 1].set_xticks(range(1, 16))
    axs[1, 1].set_xticklabels(df.columns[top_features_indices], rotation=90)
    axs[1, 1].set_title("Top 15 Feature Selection Frequency")
    axs[1, 1].set_ylabel("Selection Count")

    plt.tight_layout()

    # Save the first plot
    if not os.path.exists("plots"):
        os.makedirs("plots")
    if user_early_stop:
        fig.savefig(f"plots/PSO Analysis with swarm_size={user_swarm_size}, particle_size={particle_size}, w={user_w}, c1={user_c1}, c2={user_c2}, seed={seed}, threshold={user_threshold}, toll={user_toll}.png")
    else:   
        fig.savefig(f"plots/PSO Analysis with swarm_size={user_swarm_size}, particle_size={particle_size}, w={user_w}, c1={user_c1}, c2={user_c2}, seed={seed}, max_iteration={user_iterations}.png")

    plt.show()

    # Create a new figure for additional plots
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 5))
    if user_early_stop:
        fig2.suptitle(f"PSO Analysis with swarm_size={user_swarm_size}, particle_size={particle_size}, w={user_w}, c1={user_c1}, c2={user_c2}, seed={seed}, threshold={user_threshold}, toll={user_toll}")
    else:
        fig2.suptitle(f"PSO Analysis with swarm_size={user_swarm_size}, particle_size={particle_size}, w={user_w}, c1={user_c1}, c2={user_c2}, seed={seed}, max_iteration={user_iterations}")
    # Plot the convergence curve of the best fitness score
    axs2[0][0].plot(best_params_dict["hist_std"], label="std", color='g')
    axs2[0][0].set_xlabel("Iterations")
    axs2[0][0].set_ylabel("Standard Deviation")
    axs2[0][0].set_title("Swarm Diversity Over Iterations of the Best Run")
    axs2[0][0].legend()

    # Plot the convergence curve of the average fitness score
    axs2[0][1].bar(range(1, len(stability_scores) + 1), stability_scores)
    axs2[0][1].set_xlabel("RUNS")
    axs2[0][1].set_xticks(range(1, len(stability_scores) + 1))
    axs2[0][1].set_xticklabels([f" {i} , {i+1}" for i in range(1, len(stability_scores) + 1)], rotation=90)
    axs2[0][1].set_ylabel("Stability Coefficient") 
    axs2[0][1].set_title("Stability Coefficient between Runs")

    # Plot Exploration vs Exploitation on the same graph
    axs2[1][0].plot(best_params_dict["hist_exploitation"], label="Exploitation ", color='r')
    axs2[1][0].plot(best_params_dict["hist_exploration"], label="Exploration ", color='b')
    axs2[1][0].set_xlabel("Iteration")
    axs2[1][0].set_ylabel("Value")
    axs2[1][0].set_title("Exploration vs Exploitation Over Time")
    axs2[1][0].legend()

    # Plot execution time over iterations
    axs2[1][1].text(0.5, 0.6, f"Memory used: {memory_used} MB", fontsize=15, ha="center", va="center")
    axs2[1][1].text(0.5, 0.4, f"Total duration: {total_duration:.2f} s", fontsize=15, ha="center", va="center")
    axs2[1][1].set_xticks([])
    axs2[1][1].set_yticks([])
    axs2[1][1].set_frame_on(True)

    plt.tight_layout()

    # Save the second plot
    if user_early_stop:
        fig2.savefig(f"plots/PSO Analysis with swarm_size={user_swarm_size}, particle_size={particle_size}, w={user_w}, c1={user_c1}, c2={user_c2}, seed={seed}, threshold={user_threshold}, toll={user_toll}_additional.png")
    else:
        fig2.savefig(f"plots/PSO Analysis with swarm_size={user_swarm_size}, particle_size={particle_size}, w={user_w}, c1={user_c1}, c2={user_c2}, seed={seed},  max_iteration={user_iterations}_additional.png")

    plt.show()


    
