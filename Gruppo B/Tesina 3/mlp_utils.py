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

def plot_learning_curves(mlp):
    # Curve di apprendimento e accuratezza per epoca
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(mlp.loss_curve_, label='Training Loss')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.title('Curva di Apprendimento')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(mlp.validation_scores_, label='Validation Accuracy')
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.title('Accuratezza per epoca')
    plt.legend()

    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    # Matrice di confusione
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.title('Matrice di Confusione')
    plt.show()

def plot_roc_curve(y_test, y_proba):
    # Curve ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curve ROC')
    plt.legend()
    plt.show()