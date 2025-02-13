import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, auc
import seaborn as sns

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
scenarios = [
    # ARCHITETTURA E ATTIVAZIONE
    ## Singolo Layer 200 neuroni
    {"hidden_layer_sizes": (200,), "activation": 'identity', "solver": 'adam', "alpha": 0.001},     #0
    {"hidden_layer_sizes": (200,), "activation": 'logistic', "solver": 'adam', "alpha": 0.001},     #1
    {"hidden_layer_sizes": (200,), "activation": 'tanh', "solver": 'adam', "alpha": 0.001},         #2
    {"hidden_layer_sizes": (200,), "activation": 'relu', "solver": 'adam', "alpha": 0.001},         #3
    ## Singolo Layer 400 neuroni
    {"hidden_layer_sizes": (400,), "activation": 'identity', "solver": 'adam', "alpha": 0.001},     #4
    {"hidden_layer_sizes": (400,), "activation": 'logistic', "solver": 'adam', "alpha": 0.001},     #5
    {"hidden_layer_sizes": (400,), "activation": 'tanh', "solver": 'adam', "alpha": 0.001},         #6
    {"hidden_layer_sizes": (400,), "activation": 'relu', "solver": 'adam', "alpha": 0.001},         #7

    ## Singolo Layer 600 neuroni
    {"hidden_layer_sizes": (600,), "activation": 'identity', "solver": 'adam', "alpha": 0.001},     #8
    {"hidden_layer_sizes": (600,), "activation": 'logistic', "solver": 'adam', "alpha": 0.001},     #9
    {"hidden_layer_sizes": (600,), "activation": 'tanh', "solver": 'adam', "alpha": 0.001},         #10
    {"hidden_layer_sizes": (600,), "activation": 'relu', "solver": 'adam', "alpha": 0.001},         #11

    ## Due Layer (400,200)
    {"hidden_layer_sizes": (400,200), "activation": 'identity', "solver": 'adam', "alpha": 0.001},  #12
    {"hidden_layer_sizes": (400,200), "activation": 'logistic', "solver": 'adam', "alpha": 0.001},  #13 
    {"hidden_layer_sizes": (400,200), "activation": 'tanh', "solver": 'adam', "alpha": 0.001},      #14
    {"hidden_layer_sizes": (400,200), "activation": 'relu', "solver": 'adam', "alpha": 0.001},      #15

    ## Due Layer (600,300)
    {"hidden_layer_sizes": (600,300), "activation": 'identity', "solver": 'adam', "alpha": 0.001},  #16
    {"hidden_layer_sizes": (600,300), "activation": 'logistic', "solver": 'adam', "alpha": 0.001},  #17
    {"hidden_layer_sizes": (600,300), "activation": 'tanh', "solver": 'adam', "alpha": 0.001},      #18
    {"hidden_layer_sizes": (600,300), "activation": 'relu', "solver": 'adam', "alpha": 0.001},      #19

    ## Due Layer (800,400)
    {"hidden_layer_sizes": (800,400), "activation": 'identity', "solver": 'adam', "alpha": 0.001},  #20
    {"hidden_layer_sizes": (800,400), "activation": 'logistic', "solver": 'adam', "alpha": 0.001},  #21
    {"hidden_layer_sizes": (800,400), "activation": 'tanh', "solver": 'adam', "alpha": 0.001},      #22
    {"hidden_layer_sizes": (800,400), "activation": 'relu', "solver": 'adam', "alpha": 0.001},      #23

    ## Tre Layer (400,200,100)
    {"hidden_layer_sizes": (400,200,100), "activation": 'identity', "solver": 'adam', "alpha": 0.001},  #24
    {"hidden_layer_sizes": (400,200,100), "activation": 'logistic', "solver": 'adam', "alpha": 0.001},  #25
    {"hidden_layer_sizes": (400,200,100), "activation": 'tanh', "solver": 'adam', "alpha": 0.001},      #26 
    {"hidden_layer_sizes": (400,200,100), "activation": 'relu', "solver": 'adam', "alpha": 0.001},      #27 

    ## Tre Layer (600,300,150)
    {"hidden_layer_sizes": (600,300,150), "activation": 'identity', "solver": 'adam', "alpha": 0.001},  #28
    {"hidden_layer_sizes": (600,300,150), "activation": 'logistic', "solver": 'adam', "alpha": 0.001},  #29 
    {"hidden_layer_sizes": (600,300,150), "activation": 'tanh', "solver": 'adam', "alpha": 0.001},      #30
    {"hidden_layer_sizes": (600,300,150), "activation": 'relu', "solver": 'adam', "alpha": 0.001},      #31

    # LEARNING RATE E OTTIMIZZAZIONE
    ## Learning Rate Init 0.0001
    {"hidden_layer_sizes": (200,), "activation": 'relu', "solver": 'adam', "alpha": 0.01, 
        "learning_rate_policy": 'constant', "learning_rate_init": 0.0001, "batch_size": 16},
    {"hidden_layer_sizes": (200,), "activation": 'relu', "solver": 'adam', "alpha": 0.01, 
        "learning_rate_policy": 'constant', "learning_rate_init": 0.0001, "batch_size": 23},
    {"hidden_layer_sizes": (200,), "activation": 'relu', "solver": 'adam', "alpha": 0.01, 
        "learning_rate_policy": 'constant', "learning_rate_init": 0.0001, "batch_size": 64}
]
params = scenarios[0]

# Loop sugli scenari

print(f"\nEseguendo scenario: {params}")
    
mlp = MLPClassifier(
    hidden_layer_sizes=params["hidden_layer_sizes"],
    activation=params["activation"],
    solver=params["solver"],
    alpha=params["alpha"],
    batch_size=params.get("batch_size", 32),
    learning_rate=params.get("learning_rate_policy", 'constant'),
    learning_rate_init=params.get("learning_rate_init", 0.001),
    max_iter=1000,
    shuffle=True,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=10,
    verbose=True
)
    
# Misurazione del tempo di training
start_time = time.time()
history = mlp.fit(X_train, y_train)
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
    
# Curve di apprendimento e accuratezza per epoca
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(mlp.loss_curve_, label=f'Training Loss')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.title(f'Curva di Apprendimento')
plt.legend()
    
plt.subplot(1,2,2)
plt.plot(mlp.validation_scores_, label=f'Validation Accuracy')
plt.xlabel('Epoche')
plt.ylabel('Accuracy')
plt.title(f'Accuratezza per epoca')
plt.legend()
    
plt.show()

# Matrice di confusione
conf_matrix = confusion_matrix(y_test, mlp.predict(X_test))
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.title(f'Matrice di Confusione')
plt.show()



# Curve ROC
fpr, tpr, _ = roc_curve(y_test, mlp.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Curve ROC')
plt.legend()
plt.show()