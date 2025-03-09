import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, auc
import seaborn as sns
from scenarios import params_selection
from mlp_utils import train_and_evaluate_mlp, plot_all

# Caricamento dataset reale
file_path = "DARWIN.csv"
df = pd.read_csv(file_path)

# Separazione features e target
X = df.drop(columns=['ID', 'class'])  # Rimuoviamo ID e target
y = (df['class'] == 'P').astype(int)  # Convertiamo la classe in binario (Paziente = 1, Controllo = 0)

# Suddivisione train/test con stratificazione
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definizione degli scenari

# Definizione dei parametri disponibili
hidden_layer_sizes = [(200,), (400,), (600,), (400,200), (600,300), (800,400), (400,200,100), (600,300,150)]
activation = ['identity', 'logistic', 'tanh', 'relu']
solver = ['adam','sdg','lbfgs']
alpha = [0.0001, 0.001, 0.01, 0.1, 0.5]
learning_rate_policy = ['constant','invscaling','adaptive']
learning_rate_init = [0.0001, 0.001, 0.01, 0.1]
batch_size = [16, 32, 64]
early_stopping = [True, False]
validation_fraction = [0.1, 0.15, 0.2]
N_iter_no_change = [5, 10, 20]


#------------------------------------------

# Loop principale per la selezione dei parametri

while True:
    print("\n=== PARAMETER CONFIGURATION ===")

    user_hidden_layer_sizes = params_selection("hidden_layer_sizes", hidden_layer_sizes)
    if user_hidden_layer_sizes is None: break

    user_activation = params_selection("activation", activation)
    if user_activation is None: break

    user_solver = params_selection("solver", solver)
    if user_solver is None: break

    user_alpha = params_selection("alpha", alpha)
    if user_alpha is None: break

    user_batch_size = params_selection("batch_size", batch_size)
    if user_batch_size is None: break

    user_learning_rate = params_selection("learning_rate_policy", learning_rate_policy)
    if user_learning_rate is None: break

    user_learning_rate_init = params_selection("learning_rate_init", learning_rate_init)
    if user_learning_rate_init is None: break

    user_early_stopping = params_selection("early_stopping", early_stopping)
    if user_early_stopping is None: break

    user_validation_fraction = params_selection("validation_fraction", validation_fraction)
    if user_validation_fraction is None: break

    user_N_iter_no_change = params_selection("N_iter_no_change", N_iter_no_change)
    if user_N_iter_no_change is None: break

    print(f"\n🚀 Running MLP with user_hidden_layer_sizes={user_hidden_layer_sizes}, user_activation={user_activation}, user_solver={user_solver}, user_alpha={user_alpha}, user_batch_size={user_batch_size}\n")

    params = {
        'hidden_layer_sizes': user_hidden_layer_sizes,
        'activation': user_activation,
        'solver': user_solver,
        'alpha': user_alpha,
        'batch_size': user_batch_size,
        'learning_rate_policy': user_learning_rate,
        'learning_rate_init': user_learning_rate_init,
        'early_stopping': user_early_stopping,
        'validation_fraction': user_validation_fraction,
        'N_iter_no_change': user_N_iter_no_change
    }

    mlp = train_and_evaluate_mlp(X_train, y_train, X_test, y_test, params)
    y_pred = mlp.predict(X_test)
    y_proba = mlp.predict_proba(X_test)
    plot_all(mlp, y_test, y_pred, y_proba)

    # Ask the user if they want to run another test
    repeat = input("Do you want to run another test? (y/n): ").lower()
    if repeat != 'y':
        print("🔚 End of execution.")
        break




