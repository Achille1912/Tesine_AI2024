import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def plot_loss_curve(loss_curve, output_path, title="Loss Curve"):

    plt.figure(figsize=(7, 5))

    epochs = range(1, len(loss_curve) + 1)

    plt.plot(epochs, loss_curve, label="Loss", color="purple", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)

    plt.xticks(epochs)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path, title="Confusion Matrix"):

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def plot_accuracy_boxplots(train_accuracies, test_accuracies, output_path):

    data = {
        "Train": train_accuracies,
        "Test": test_accuracies
    }

    plt.figure(figsize=(8, 6))

    sns.boxplot(data=data, palette=["skyblue", "salmon"])

    sns.swarmplot(data=data, color=".25", alpha=0.6)

    mean_train = np.mean(train_accuracies)
    mean_test = np.mean(test_accuracies)

    plt.text(0, mean_train, f"{mean_train:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='blue')
    plt.text(1, mean_test, f"{mean_test:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')

    median_train = np.median(train_accuracies)
    median_test = np.median(test_accuracies)
    plt.title(f"Accuracy Distribution - : Median Train {median_train:.2f}, Median Test: {median_test:.2f}")

    plt.ylabel("Accuracy")
    plt.xticks([0, 1], ["Train", "Test"])

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(output_path)
    plt.close()


def plot_mean_roc_curve(fprs_list, tprs_list, aucs_list, output_path=None, num_runs=None):
    """
    Plot mean ROC curve over multiple runs.
    """

    mean_fpr = np.linspace(0, 1, 100)

    tprs_interpolated = []
    for fpr, tpr in zip(fprs_list, tprs_list):
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs_interpolated.append(tpr_interp)

    mean_tpr = np.mean(tprs_interpolated, axis=0)
    std_tpr = np.std(tprs_interpolated, axis=0)

    mean_auc = np.mean(aucs_list)
    std_auc = np.std(aucs_list)

    plt.figure(figsize=(8, 6))

    plt.plot(
        mean_fpr,
        mean_tpr,
        label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
        lw=2
    )

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr,
        tpr_lower,
        tpr_upper,
        color='gray',
        alpha=0.2,
        label="±1 Std. Dev."
    )

    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')

    title = f"Mean ROC Curve ({num_runs} runs)" if num_runs else "Mean ROC Curve"
    plt.title(title)
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.grid(True)

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC plot salvato in: {output_path}")
    else:
        plt.show()

    plt.close()

