from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Dict, Any


def get_all_classifiers() -> Dict[str, Any]:
    """
    Get dictionary of all classifiers
    :return: dict
        Dictionary with classifier names as keys and
        (classifier, param_grid) tuples as values
    """
    classifiers = {
        "logistic_regression": (
            LogisticRegression(),
            {
                "C": [1e-3, 1e-2, 1e-1, 1, 10, 100],
                "solver": ["lbfgs", "liblinear"],
            }
        ),
        "swm_linear": (
            SVC(kernel="linear", probability=True),
            {
                "C": [0.1, 1, 10, 100],
            }
        ),
        "swm_rbf": (
            SVC(kernel="rbf", probability=True),
            {
                "C": [0.1, 1, 10, 100],
                "gamma": [0.001, 0.01, 0.1, 1],
            }
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            }
        ),
        "decision_tree": (
            DecisionTreeClassifier(),
            {
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"],
            }
        ),
        "random_forest": (
            RandomForestClassifier(),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
            }
        )
    }
    return classifiers

def get_classifier_by_name(name: str) -> tuple[Any, Dict[str, list]]:
    """
    Get classifier by name

    :param name: str
        Name of classifier
    :return: tuple (classifier, param_grid)
    :raises: KeyError
        If classifier name doesn't exist
    """
    classifiers = get_all_classifiers()
    if name not in classifiers:
        available_classifiers = list(classifiers.keys())
        raise KeyError(
            f"Classifier with name '{name}' not found"
            f"Available classifiers: {available_classifiers}"
        )

    return classifiers[name]
