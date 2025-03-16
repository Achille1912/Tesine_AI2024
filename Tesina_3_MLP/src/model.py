from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from .custom_mlp import MLPDropout
from sklearn.model_selection import StratifiedKFold
import numpy as np


def train_model(X_train, y_train):
    """
    Trains a Multi-Layer Perceptron (MLP) model using the given training data and
    configures various hyperparameters for model instantiation. Returns the trained
    model, the loss curve over training iterations, and the evaluation accuracy on
    the training set.

    :param X_train: Training feature dataset.
    :type X_train: ndarray
    :param y_train: Training target labels corresponding to the feature dataset.
    :type y_train: ndarray
    :return: A tuple containing the following:

        1. Trained MLP model.
        2. Loss curve recorded during training.
        3. Accuracy of the model evaluated on the training dataset.

    :rtype: tuple
    """

    # Model instantiation
    mlp = MLPClassifier(
        hidden_layer_sizes = (400, 200), # [(200), (400), (600), (400,200), (600,300), (800,400), (400,200,100), (600,300,150)]
        activation = 'tanh',             # ['identity', 'logistic', 'tanh', 'relu']
        solver = 'adam',                 # ['adam','sgd','lbfgs']
        alpha = 0.001,                   # [0.0001, 0.001, 0.01, 0.1, 0.5]
        batch_size = 16,                 # [16, 32, 64]
        max_iter = 1000,
        random_state = 42,
        learning_rate = 'adaptive',      # ['constant', 'invscaling', 'adaptive']
        learning_rate_init = 0.1,        # [0.0001, 0.001, 0.01, 0.1]
        validation_fraction = 0.1,       # [0.1, 0.15, 0.2]
        n_iter_no_change = 10,           # [5, 10, 20]
        early_stopping = True,
        shuffle = True,
        verbose = True
    )


    '''
    # Model instantiation
    mlp = MLPDropout(
        hidden_layer_sizes = (400),  #  [(200), (400), (600), (400,200), (600,300), (800,400), (400,200,100), (600,300,150)]
        activation = 'tanh',         #  ['identity', 'logistic', 'tanh', 'relu']
        solver = 'adam',             #  ['adam','sgd','lbfgs']
        alpha = 0.01,                #  [0.0001, 0.001, 0.01, 0.1, 0.5]
        batch_size = 8,              #  [16, 32, 64]
        max_iter = 1000,
        random_state = 42,
        learning_rate = 'adaptive',  # ['constant', 'invscaling', 'adaptive']
        learning_rate_init = 0.001,  # [0.0001, 0.001, 0.01, 0.1]
        validation_fraction = 0.15 , # [0.1, 0.15, 0.2]
        n_iter_no_change = 20 ,      # [5, 10, 20]
        dropout = 0.3,
        early_stopping = True,
        shuffle = True,
        verbose = True
    )
    '''

    mlp.fit(X_train, y_train)

    # Compute accuracy on training set
    train_accuracy = mlp.score(X_train, y_train)

    # Scikit-learn saves the loss at each iteration in mlp.loss_curve_
    train_loss_curve = mlp.loss_curve_

    return mlp, train_loss_curve, train_accuracy


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the accuracy of a given model on a test dataset.

    This function uses the provided pre-trained machine learning model to predict
    the target values of the test dataset. It then calculates the accuracy score
    by comparing the predicted values with the actual target values from the test
    data. The function returns the calculated accuracy score as a float value.

    :param model: A pre-trained machine learning model.
    :param X_test: Test dataset features.
    :param y_test: Actual target values corresponding to the test dataset.
    :return: The accuracy score of the model on the test dataset.
    :rtype: float
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def cross_validate_model(X, y, n_splits=5):
    """
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_accuracies = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}:")
        
        X_train_fold = X[train_index]
        X_test_fold = X[test_index]
        
        y_train_fold = y.iloc[train_index]
        y_test_fold = y.iloc[test_index]

        # Train the model on the current fold
        model, train_loss_curve, train_accuracy = train_model(X_train_fold, y_train_fold)

        # Evaluate the model on the current fold test
        test_accuracy = evaluate_model(model, X_test_fold, y_test_fold)
        cv_accuracies.append(test_accuracy)

        print(f"  Accuracy Training: {train_accuracy:.4f} - Accuracy Test: {test_accuracy:.4f}\n")

    mean_accuracy = np.mean(cv_accuracies)
    print("Accuratezza media sui test fold: {:.4f}".format(mean_accuracy))

    return cv_accuracies



