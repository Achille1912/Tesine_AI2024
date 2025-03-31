import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.data_conf import DataConfig


def load_and_preprocess_data(config: DataConfig):
    """
    Loads and preprocesses the dataset specified in the configuration.
    """

    logging.info(" Dataset loading...")
    df = pd.read_csv(config.dataset_path)

    X = df.drop(columns=['ID', 'class'])
    y = (df['class'] == 'P').astype(int)  # 'P' -> 1, 'C' -> 0

    num_features_before = X.shape[1]  

    logging.info(f"Original dataset: {df.shape[0]} samples, {num_features_before} features.")

    X_train, X_test, y_train, y_test = train_test_split( X,
                                                         y,
                                                         test_size=config.test_size,
                                                         random_state=config.random_state,
                                                         stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    plot_feature_distribution(X_train_scaled, y_train, title="Features Distribution (Before PCA)")

    pca = PCA(n_components=config.n_components, random_state=config.random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    num_features_after = X_train_pca.shape[1]  

    variance_explained = np.sum(pca.explained_variance_ratio_) * 100
    logging.info(f" PCA applied: Reduction from {num_features_before} to {num_features_after} features.")
    logging.info(f" Total variance explained by PCA: {variance_explained:.2f}%")

    plot_feature_distribution(X_train_pca, y_train, title="Features Distribution (After PCA)", pca_model=pca)

    return X_train_pca, X_test_pca, y_train, y_test


def print_data_summary(X_train, X_test, y_train, y_test):
    """
    Log summary statistics about the dataset.
    """

    train_samples = X_train.shape[0]
    test_samples = X_test.shape[0]
    num_features = X_train.shape[1]

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    class_distribution_train = {u: f"{c} samples ({c / train_samples * 100:.2f}%)" for u, c in zip(unique_train, counts_train)}

    unique_test, counts_test = np.unique(y_test, return_counts=True)
    class_distribution_test = {u: f"{c} samples ({c / test_samples * 100:.2f}%)" for u, c in zip(unique_test, counts_test)}

    summary_message = (
        f"\n **Dataset Summary**\n"
        f"--------------------------\n"
        f" Training samples: {train_samples}\n"
        f" Testing samples: {test_samples}\n"
        f" Number of features: {num_features}\n\n"
        f" **Class distribution (Training set):**\n"
        + "\n".join([f"    Class {k}: {v}" for k, v in class_distribution_train.items()]) +
        f"\n\n **Class distribution (Test set):**\n"
        + "\n".join([f"    Class {k}: {v}" for k, v in class_distribution_test.items()])
    )

    logging.info(summary_message)


def plot_feature_distribution(X, y, title, pca_model=None):
    """
    Plot the feature distribution before and after PCA.
    """

    fig = plt.figure(figsize=(10, 7), dpi=150)

    custom_palette = {0: "blue", 1: "red"}

    if pca_model:
        if X.shape[1] < 2:
            raise ValueError("Error: Dataset has less than 2 principal components, unable to generate PCA plot.")

        explained_variance_pc1_pc2 = pca_model.explained_variance_ratio_[:2].sum() * 100
        total_variance = pca_model.explained_variance_ratio_.sum() * 100

        title += f"\n(PC1+PC2 explain the {explained_variance_pc1_pc2:.2f}% of total variance {total_variance:.2f}%)"

        sns.scatterplot(x=X[:, 0],
                        y=X[:, 1],
                        hue=y,
                        palette=custom_palette,
                        alpha=0.7,
                        s=60,
                        edgecolor="black")
        plt.xlabel("PC1 (First Principal Component)")
        plt.ylabel("PC2 (Second Principal Component)")

    else:
        variances = np.var(X, axis=0)
        top_features = np.argsort(variances)[-2:]

        sns.scatterplot(x=X[:, top_features[0]],
                        y=X[:, top_features[1]],
                        hue=y,
                        palette=custom_palette,
                        alpha=0.7,
                        s=60,
                        edgecolor="black")
        plt.xlabel(f"Feature {top_features[0]} (Highest variance)")
        plt.ylabel(f"Feature {top_features[1]} (Second highest variance)")

    plt.title(title)
    plt.grid(True)
    plt.legend(title="Classe")
    plt.show()
