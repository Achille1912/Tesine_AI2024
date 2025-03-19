import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np



def plot_loss_curve(loss_curve, output_path=None):
    """
    Plots a training loss curve based on the provided loss values across epochs. This
    function either displays the plot or saves it to a specified file path.

    :param loss_curve: A list or array-like object containing the loss values for
        each epoch.
    :param output_path: Optional string specifying the file path to save the plot
        image. If not provided, the plot is displayed instead of being saved.

    :return: None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(loss_curve, marker='o', linestyle='-')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, output_path=None):
    """
    Generates and displays the confusion matrix for predictions made by a given model.

    This function computes the confusion matrix for the provided true labels and predictions
    obtained using the given model. It then visualizes the matrix using a plot and either
    displays it interactively or saves it to a specified file path.

    :param model: The trained predictive model implementing the `predict` method.
    :type model: object
    :param X_test: Data used to generate predictions with the model.
    :type X_test: array-like
    :param y_test: True labels for comparison with predictions.
    :type y_test: array-like
    :param output_path: Optional file path to save the generated confusion matrix plot.
                        If None, the plot is displayed interactively.
    :type output_path: str, optional
    :return: None
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def plot_roc_curve(model, X_test, y_test, output_path):
    """
    Displays the ROC curve and calculates the AUC for the given model.
    """

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def boxplot_scores(scores):
    """
    Displays a boxplot of scores.
    
    :param scores: A list or array of accuracy values or other performance metrics.
    :type scores: list or array-like
    :return: None
    """
    plt.figure()
    plt.boxplot(scores)
    plt.title("Boxplot of Scores")
    plt.ylabel("Accuracy")

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



