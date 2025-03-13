import sys

sys.dont_write_bytecode = True

import os
import numpy as np
import time
import logging
from datetime import datetime

from src.config import DataConfig
from src.data import load_and_preprocess_data
from src.model import train_model, evaluate_model, cross_validate_model
from utils.visualization import plot_loss_curve, plot_confusion_matrix, plot_roc_curve, boxplot_scores


def create_output_directory():
    """Create an output folder with a timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(log_file_path):
    """
    Sets up logging for the application by configuring the root logger with a file
    handler. The function clears any existing handlers from the root logger, 
    ensures log messages are formatted with timestamps, log levels, and log 
    messages.

    :param log_file_path: The file path where log messages will be written.
    :type log_file_path: str
    :return: None
    """
    logger = logging.getLogger()  # Recover root logger
    logger.setLevel(logging.INFO)


    if logger.hasHandlers():
        logger.handlers.clear()

    # Create an handler for the file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Msg format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)


def print_data_summary(X_train, X_test, y_train, y_test):
    """Log summary statistics about the dataset."""
    logging.info("\nData Summary:")
    logging.info(f"Training samples: {X_train.shape[0]}")
    logging.info(f"Testing samples: {X_test.shape[0]}")
    logging.info(f"Number of features: {X_train.shape[1]}")
    logging.info("Class distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        perc = c / len(y_train) * 100
        logging.info(f"  Class {u}: {c} samples ({perc:.2f}%)")


def main():

    output_dir = create_output_directory()

    log_file = os.path.join(output_dir, "training.log")
    setup_logging(log_file)

    logging.info("=== main.py start ===")

    config = DataConfig(
        dataset_path="DARWIN.csv",
        test_size=0.2,  # choose 0.1 for MLP with dropout
        random_state=42,
        n_components=0.97
    )

    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    print_data_summary(X_train, X_test, y_train, y_test)

    # Model training
    start_time = time.time()
    model, train_loss_curve, train_accuracy = train_model(X_train, y_train)
    train_time = time.time() - start_time

    logging.info(f"Training completed in {train_time:.2f} seconds.")
    logging.info(f"Training accuracy: {train_accuracy:.4f}")

    loss_plot_path = os.path.join(output_dir, "loss_curve.png")
    plot_loss_curve(train_loss_curve, output_path=loss_plot_path)
    logging.info(f"Loss curve plot saved to {loss_plot_path}")

    # Model evaluation
    test_accuracy = evaluate_model(model, X_test, y_test)
    logging.info(f"Test accuracy: {test_accuracy:.4f}")

    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(model, X_test, y_test, output_path=cm_plot_path)
    logging.info(f"Confusion matrix plot saved to {cm_plot_path}")


    plot_roc_curve(model, X_test, y_test)

    boxplot_scores(scores=[train_accuracy, test_accuracy])

    # Cross validation
    logging.info("=== Starting of Cross Validation on training set ===")
    cv_accuracies = cross_validate_model(X_train, y_train, n_splits=5)
    mean_cv_accuracy = np.mean(cv_accuracies)
    logging.info("Cross Validation accuracies per fold: " + ", ".join(f"{acc:.4f}" for acc in cv_accuracies))
    logging.info(f"Mean Cross Validation accuracy: {mean_cv_accuracy:.4f}")

    logging.info("=== main.py end ===")


if __name__ == "__main__":
    main()
