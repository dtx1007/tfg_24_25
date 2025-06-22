import os
import pickle
import tempfile
import time
import numpy as np
import xgboost as xgb

from dataclasses import dataclass, asdict
from typing import Literal, Any

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)


@dataclass(frozen=True)
class MLHyperparams:
    """
    Dataclass to hold hyperparameters for the classic machine learning models.

    Parameters
    ----------
    model_type : str
        Type of model to create ('random_forest', 'knn', 'decision_tree', 'xgboost',
        'svm', 'gradient_boosting', 'logistic_regression', 'naive_bayes')
    n_estimators : int
        Number of estimators for ensemble models (Random Forest, Gradient Boosting, XGBoost)
    max_depth : int or None
        Maximum depth of trees for tree-based models
    min_samples_split : int
        Minimum samples required to split a node for tree-based models
    min_samples_leaf : int
        Minimum samples required at a leaf node for tree-based models
    class_weight : str or dict or None
        Class weights for handling imbalanced datasets
    C : float
        Regularization parameter for SVM and Logistic Regression
    kernel : str
        Kernel type for SVM
    gamma : str or float
        Kernel coefficient for SVM with RBF, poly or sigmoid kernel
    probability : bool
        Whether to enable probability estimates for SVM
    solver : str
        Algorithm to use for Logistic Regression optimization
    n_neighbors : int
        Number of neighbors for KNN
    weights : str
        Weight function for KNN
    learning_rate : float
        Learning rate for XGBoost
    subsample : float
        Subsample ratio for XGBoost
    colsample_bytree : float
        Column subsample ratio for XGBoost
    random_state : int
        Seed for reproducibility
    verbose : bool
        Whether to print verbose output during model training
    """

    model_type: Literal[
        "random_forest",
        "knn",
        "decision_tree",
        "xgboost",
        "svm",
        "gradient_boosting",
        "logistic_regression",
        "naive_bayes",
    ] = "random_forest"

    # Common parameters
    random_state: int = 42
    class_weight: Literal["balanced", "balanced_subsample"] | dict | None = "balanced"

    # Random Forest, Gradient Boosting, XGBoost, Decision Tree parameters
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1

    # KNN parameters
    n_neighbors: int = 5
    weights: Literal["uniform", "distance"] = "uniform"

    # XGBoost specific parameters
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0  # SVM parameters
    C: float = 1.0
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
    gamma: Literal["scale", "auto"] | float = "scale"
    probability: bool = True

    # Logistic Regression parameters
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "lbfgs"

    # Verbose output for some models
    verbose: bool = False

    def create_model(self) -> Any:
        """
        Create and return a machine learning model based on the hyperparameters.

        Returns
        -------
        model : sklearn estimator
            The initialized model according to the specified hyperparameters
        """

        match self.model_type:
            case "random_forest":
                return RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    verbose=self.verbose,
                )
            case "gradient_boosting":
                return GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                    verbose=self.verbose,
                )
            case "decision_tree":
                return DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                )
            case "knn":
                return KNeighborsClassifier(
                    n_neighbors=self.n_neighbors,
                    weights=self.weights,
                )
            case "xgboost":
                return xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth if self.max_depth is not None else 6,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    random_state=self.random_state,
                )
            case "svm":
                return SVC(
                    C=self.C,
                    kernel=self.kernel,
                    gamma=self.gamma,
                    probability=self.probability,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    verbose=self.verbose,
                )
            case "logistic_regression":
                return LogisticRegression(
                    C=self.C,
                    class_weight=self.class_weight,
                    solver=self.solver,
                    random_state=self.random_state,
                    verbose=self.verbose,
                )
            case "naive_bayes":
                return GaussianNB()
            case _:
                raise ValueError(f"Unknown model type: {self.model_type}")


def train_classical_models_cv(
    hyperparams_dict: dict[str, MLHyperparams] | None = None,
    models_dict: dict[str, Any] | None = None,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    n_folds: int = 5,
    n_repetitions: int = 3,
    scoring_metric: Literal[
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    ] = "f1",
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Train and evaluate classical machine learning models using repeated stratified k-fold cross-validation.
    Returns the evaluation results and the best model instances.

    This function allows for two ways of providing models:
    1. By providing a dictionary of MLHyperparams objects
    2. By providing pre-instantiated model objects directly

    Parameters
    ----------
    hyperparams_dict : dict, optional
        Dictionary mapping model names to MLHyperparams objects.
        If provided, models will be created from these hyperparameters.
    models_dict : dict, optional
        Dictionary mapping model names to pre-instantiated sklearn model objects.
        Only used if hyperparams_dict is None.
    X : numpy.ndarray
        Feature matrix [n_samples, n_features]
    y : numpy.ndarray
        Target labels [n_samples]
    n_folds : int
        Number of folds for stratified k-fold cross-validation
    n_repetitions : int
        Number of times to repeat the k-fold cross-validation with different splits
    scoring_metric : str
        Primary metric to use for model evaluation
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (results, best_models)
        - results: Nested dictionary containing evaluation results for each model
        - best_models: Dictionary mapping model names to their best performing instances
    """

    # Validate inputs
    if X is None or y is None:
        raise ValueError("X and y must be provided")

    if hyperparams_dict is None and models_dict is None:
        raise ValueError(
            "Either hyperparams_dict or models_dict must be provided"
        )  # Set up models dictionary
    if hyperparams_dict is not None:
        # Create models from hyperparameters (preferred approach)
        models = {
            name: params.create_model() for name, params in hyperparams_dict.items()
        }
        # Save hyperparameters for results
        hyperparams_record = {
            name: asdict(params) for name, params in hyperparams_dict.items()
        }
    else:
        # Use pre-instantiated models
        models = models_dict
        hyperparams_record = None

    # Setup cross-validation
    cv_splitter = RepeatedStratifiedKFold(
        n_repeats=n_repetitions, n_splits=n_folds, random_state=random_state
    )

    # Initialize results dictionary
    results = {}
    # Dictionary to track best models and their scores
    best_models = {}
    best_scores = {}

    for model_name in models:
        results[model_name] = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "pr_auc": [],
            "roc_auc": [],
            "pr_curve_data": [],
            "roc_curve_data": [],
            "conf_matrices": [],
            "model_size": [],
            "training_time": [],
            "hyperparams": hyperparams_record.get(model_name)
            if hyperparams_record
            else None,
        }
        # Initialize best score tracking for each model
        best_scores[model_name] = -np.inf

    print(
        f"Training {len(models)} models with {n_repetitions} x {n_folds}-fold cross-validation..."
    )

    # Perform cross-validation
    fold_count = 1
    total_folds = n_repetitions * n_folds

    for rep_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        # Calculate current repetition and fold
        current_rep = (rep_idx // n_folds) + 1
        current_fold = (rep_idx % n_folds) + 1

        print(
            f"\n=== Repetition {current_rep}/{n_repetitions}, Fold {current_fold}/{n_folds} ==="
        )  # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Print class distribution
        unique_classes, train_counts = np.unique(y_train, return_counts=True)
        test_counts = [np.sum(y_test == c) for c in unique_classes]

        print("Class distribution:")
        for i, c in enumerate(unique_classes):
            print(f"  Class {c}: {train_counts[i]} train, {test_counts[i]} test")

        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"Training {model_name}...")

            # Clone the model to ensure independent evaluation in each fold
            from sklearn.base import clone

            model_clone = clone(model)

            # Train the model and measure time
            start_time = time.time()
            model_clone.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Generate predictions
            y_pred = model_clone.predict(X_test)

            # Get probability predictions or decision function values
            if hasattr(model_clone, "predict_proba"):
                y_prob = model_clone.predict_proba(X_test)[:, 1]
            elif hasattr(model_clone, "decision_function"):
                decision_scores = model_clone.decision_function(X_test)
                # Normalize to 0-1 range for metrics
                y_prob = (decision_scores - decision_scores.min()) / (
                    decision_scores.max() - decision_scores.min() + 1e-10
                )
            else:
                y_prob = y_pred  # Fallback, but AUC metrics won't be meaningful

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # Precision-Recall Curve
            pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc_score = auc(pr_recall, pr_precision)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc_score = auc(fpr, tpr)

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Calculate model size
            fd, temp_path = tempfile.mkstemp()
            os.close(fd)
            pickle.dump(model_clone, open(temp_path, "wb"))
            model_size_kb = os.path.getsize(temp_path) / 1024  # Size in KB
            os.remove(temp_path)

            # Store results
            results[model_name]["accuracy"].append(accuracy)
            results[model_name]["precision"].append(precision)
            results[model_name]["recall"].append(recall)
            results[model_name]["f1"].append(f1)
            results[model_name]["pr_auc"].append(pr_auc_score)
            results[model_name]["roc_auc"].append(roc_auc_score)
            results[model_name]["pr_curve_data"].append((pr_precision, pr_recall))
            results[model_name]["roc_curve_data"].append((fpr, tpr))
            results[model_name]["conf_matrices"].append(conf_matrix)
            results[model_name]["model_size"].append(model_size_kb)
            results[model_name]["training_time"].append(training_time)

            print(
                f"  {model_name}: Acc={accuracy:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, ROC: {roc_auc_score:.4f}, PR: {pr_auc_score:.4f}, Size={model_size_kb:.1f}KB, Time={training_time:.2f}s"
            )

            # Track best model based on scoring metric
            current_score = locals()[scoring_metric]  # Get metric value by name
            if current_score > best_scores[model_name]:
                best_scores[model_name] = current_score
                # Save a copy of this model as it's the best so far
                best_models[model_name] = pickle.loads(pickle.dumps(model_clone))
                print(
                    f"  ★ New best {model_name} model: {scoring_metric}={current_score:.4f}"
                )

        fold_count += 1  # Calculate summary statistics for each model
    for model_name in models:
        model_results = results[model_name]

        # Calculate means and standard deviations for all metrics
        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
            "model_size",
            "training_time",
        ]

        for metric in metric_names:
            model_results[f"mean_{metric}"] = np.mean(
                model_results[metric], dtype=np.float64
            )
            model_results[f"std_{metric}"] = np.std(
                model_results[metric], dtype=np.float64
            )

        # Find the best fold based on the specified scoring metric
        best_fold_idx = np.argmax(model_results[scoring_metric])
        model_results["best_fold_idx"] = int(best_fold_idx)

        # Store the best score achieved for this model
        model_results["best_score"] = best_scores[model_name]

        print(f"\n=== {model_name} Summary ===")
        print(
            f"Mean {scoring_metric.upper()}: {model_results[f'mean_{scoring_metric}']:.4f} ± {model_results[f'std_{scoring_metric}']:.4f}"
        )
        print(f"Best {scoring_metric.upper()}: {best_scores[model_name]:.4f}")
        print(f"Mean Size: {model_results['mean_model_size']:.1f}KB")
        print(f"Mean Training Time: {model_results['mean_training_time']:.2f}s")

    print("\nBest models saved for each model type.")

    return results, best_models
