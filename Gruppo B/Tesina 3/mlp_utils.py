import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, auc

def train_and_evaluate_mlp(X_train, y_train, X_test, y_test, params):
    mlp = MLPClassifier(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        activation=params['activation'],
        solver=params['solver'],
        alpha=params['alpha'],
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate_policy'],
        learning_rate_init=params['learning_rate_init'],
        max_iter=1000,
        shuffle=True,
        random_state=42,
        early_stopping=params['early_stopping'],
        validation_fraction=params['validation_fraction'],
        n_iter_no_change=params['N_iter_no_change'],
        verbose=True
    )

    # Misurazione del tempo di training
    start_time = time.time()
    mlp.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    # Valutazione del modello
    train_accuracy = accuracy_score(y_train, mlp.predict(X_train))
    test_accuracy = accuracy_score(y_test, mlp.predict(X_test))
    train_loss = log_loss(y_train, mlp.predict_proba(X_train))
    test_loss = log_loss(y_test, mlp.predict_proba(X_test))

    print(f'Tempo di training: {training_time:.2f} secondi')
    print(f'Accuracy Train: {train_accuracy:.4f}, Accuracy Test: {test_accuracy:.4f}')
    print(f'Loss Train: {train_loss:.4f}, Loss Test: {test_loss:.4f}')

    return mlp

def plot_all(mlp, y_test, y_pred, y_proba):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Curve di apprendimento
    axes[0, 0].plot(mlp.loss_curve_, label='Training Loss')
    axes[0, 0].set_xlabel('Epoche')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Curva di Apprendimento')
    axes[0, 0].legend()

    # Accuratezza per epoca
    axes[0, 1].plot(mlp.validation_scores_, label='Validation Accuracy')
    axes[0, 1].plot(mlp.loss_curve_, label='Training Accuracy')
    axes[0, 1].set_xlabel('Epoche')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuratezza per epoca')
    axes[0, 1].legend()

    # Matrice di confusione
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predetto')
    axes[1, 0].set_ylabel('Reale')
    axes[1, 0].set_title('Matrice di Confusione')

    # Curve ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    axes[1, 1].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[1, 1].plot([0, 1], [0, 1], linestyle='--')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('Curve ROC')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()