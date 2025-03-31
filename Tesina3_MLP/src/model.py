import logging
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix


class MLPExperiment:
    """
    A class that encapsulates an MLPClassifier and provides methods for
    training, evaluation, and cross_validation.
    """

    def __init__(
        self,
        hidden_layer_sizes,
        activation,
        solver,
        alpha,
        batch_size,
        max_iter,
        random_state,
        learning_rate,
        learning_rate_init,
        validation_fraction,
        n_iter_no_change,
        early_stopping,
        shuffle,
        verbose
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state

        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping = early_stopping
        self.shuffle = shuffle
        self.verbose = verbose

        # Build the scikit-learn model
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            early_stopping=self.early_stopping,
            shuffle=self.shuffle,
            verbose=self.verbose
        )

        logging.info(
            f"Training MLP with random_state={self.random_state}, "
            f"hidden_layer_sizes={self.hidden_layer_sizes}, activation={self.activation}, "
            f"solver={self.solver}, alpha={self.alpha}, "
            f"learning_rate={self.learning_rate}, learning_rate_init={self.learning_rate_init}, "
            f"validation_fraction={self.validation_fraction}, n_iter_no_change={self.n_iter_no_change}, "
            f"early_stopping={self.early_stopping}, shuffle={self.shuffle}, verbose={self.verbose}"
        )


    def train(self, X_train, y_train):

        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()

        train_time = end_time - start_time

        y_pred_train = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)

        loss_curve = self.model.loss_curve_

        return train_time, train_accuracy, loss_curve


    def evaluate(self, X_test, y_test):

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)

        return acc, cm


    def cross_validate(self, X_train, y_train, folds=5):

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        cv_accuracies = []

        for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train), start=1):
            
            logging.info(f"    Fold {fold}/{folds} in progress...")

            X_train_fold = X_train[train_index]
            X_test_fold = X_train[test_index]

            y_train_fold = y_train.iloc[train_index]
            y_test_fold = y_train.iloc[test_index]

            self.model.fit(X_train_fold, y_train_fold)

            y_pred  = self.model.predict(X_test_fold)
            cv_accuracy = np.mean(y_pred == y_test_fold)
            cv_accuracies.append(cv_accuracy)

            logging.info(f"    Fold {fold}/{folds} accuracy = {cv_accuracy:.4f}")

        return cv_accuracies


    def single_run(self, X_train, y_train, X_test, y_test, folds_cv=5):
        """
        """
        train_time, train_acc, loss_curve  = self.train(X_train, y_train)

        test_acc, conf_matrix  = self.evaluate(X_test, y_test)

        cv_scores = []
        cv_mean = None
        if folds_cv > 1:
            cv_scores = self.cross_validate(X_train,
                                            y_train,
                                            folds=folds_cv)
            cv_mean = np.mean(cv_scores)
            logging.info(f" Cross-validation folds={folds_cv}: mean_acc={cv_mean:.4f}")

        fpr_test, tpr_test, roc_auc_test = self.compute_roc_auc(X_test,
                                                                y_test)

        result = {
            "train_time": train_time,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "confusion_matrix": conf_matrix,
            "loss_curve": loss_curve,
            "cv_scores": cv_scores,
            "cv_mean_accuracy": cv_mean,
            "fpr_test": fpr_test,
            "tpr_test": tpr_test,
            "roc_auc_test": roc_auc_test
        }

        return result


    def compute_roc_auc(self, X_test, y_test):

        def calculate_roc_metrics(y_true, y_proba):
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            return fpr, tpr, auc(fpr, tpr)

        y_test_proba = self.model.predict_proba(X_test)[:, 1]

        fpr_test, tpr_test, roc_auc_test = calculate_roc_metrics(y_test, y_test_proba)

        return fpr_test, tpr_test, roc_auc_test
