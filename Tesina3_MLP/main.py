import json
import sys
import os
import numpy as np
import logging
from src.data_conf import DataConfig
from src.data_processing import load_and_preprocess_data, print_data_summary
from src.experiment_config import get_user_config
from src.model import MLPExperiment
from utils.logging_setup import create_output_directory, setup_logging
from utils.visualization import plot_accuracy_boxplots, plot_confusion_matrix, plot_loss_curve, plot_mean_roc_curve

sys.dont_write_bytecode = True


NUM_RUNS = 30

def main():

   output_dir = create_output_directory()
   main_log = os.path.join(output_dir, "log.txt")

   setup_logging(main_log)

   logging.info("=== Start experiment with 30 runs ===")

   user_config = get_user_config()  
   config_path = os.path.join(output_dir, "config.json")

   with open(config_path, "w") as f:
       json.dump(user_config, f, indent=4)
   logging.info(f"Configurazione salvata in: {config_path}")


   individual_runs_dir = os.path.join(output_dir, "individual_runs")
   os.makedirs(individual_runs_dir, exist_ok=True)


   test_accuracies = []
   train_accuracies = []

   all_fprs_test = []
   all_tprs_test = []
   all_aucs_test = []

   cv_means_all_runs = []

   for i in range(NUM_RUNS):

       run_dir = os.path.join(individual_runs_dir, f"run_{i + 1}")  
       os.makedirs(run_dir, exist_ok=True)
       run_log_file = os.path.join(run_dir, f"run_{i+1}.log")

       logging.info(f"--- Avvio run {i + 1}/{NUM_RUNS} con random_state={42+i} ---")

       setup_logging(run_log_file, console_output=False)


       data_config = DataConfig(
           dataset_path="DARWIN.csv",
           test_size=0.2,
           random_state=42 + i,
           n_components=0.97
       )

       X_train, X_test, y_train, y_test = load_and_preprocess_data(data_config)

       print_data_summary(X_train, X_test, y_train, y_test)

       mlp = MLPExperiment(
           hidden_layer_sizes=user_config["hidden_layer_sizes"],
           activation=user_config["activation"],
           solver=user_config["solver"],
           alpha=user_config["alpha"],
           batch_size=user_config["batch_size"],
           max_iter=1000,
           random_state= 42 + i,  
           learning_rate = user_config["learning_rate_policy"],
           learning_rate_init = user_config["learning_rate_init"],
           validation_fraction = user_config["validation_fraction"],
           n_iter_no_change = user_config["n_iter_no_change"],
           early_stopping = True,
           shuffle= True,
           verbose= True
       )

     
       results = mlp.single_run(X_train, y_train, X_test, y_test, folds_cv=5)


       if results["cv_mean_accuracy"] is not None:
           cv_means_all_runs.append(results["cv_mean_accuracy"])
       else:
           logging.info("Nessuna cross-validation eseguita in questa run.")

     
       test_accuracies.append(results["test_accuracy"])
       train_accuracies.append(results["train_accuracy"])

       all_fprs_test.append(results["fpr_test"])
       all_tprs_test.append(results["tpr_test"])
       all_aucs_test.append(results["roc_auc_test"])


       loss_output_path = os.path.join(run_dir, "loss_curve.png")
       plot_loss_curve(results["loss_curve"], output_path=loss_output_path)

       conf_matrix = results["confusion_matrix"]
       cm_output_path = os.path.join(run_dir, "confusion_matrix.png")
       plot_confusion_matrix(conf_matrix, class_names=["C", "P"], output_path=cm_output_path)

       logging.info(f"=== Run {i + 1}/{NUM_RUNS} results with seed={42+i} ===")

       setup_logging(main_log, console_output=False)


  
   plot_accuracy_boxplots(train_accuracies, test_accuracies, os.path.join(output_dir, "accuracy_boxplot.png"))

   plot_mean_roc_curve(
       all_fprs_test,
       all_tprs_test,
       all_aucs_test,
       os.path.join(output_dir, "mean_roc_curve_runs.png"),
       NUM_RUNS
   )


  
   if len(cv_means_all_runs) > 0:
       final_cv_mean = np.mean(cv_means_all_runs)
       final_cv_std = np.std(cv_means_all_runs)

      
       logging.info(f"=== CV Mean Accuracy su {NUM_RUNS} run: {final_cv_mean:.4f} +/- {final_cv_std:.4f} ===")



if __name__ == "__main__":
    main()
