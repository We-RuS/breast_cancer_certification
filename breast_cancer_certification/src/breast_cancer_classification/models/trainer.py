import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator
from typing import Dict, Any, Tuple, List
import warnings

from .classifier import get_all_classifiers
from ..data.preprocessor import DataPreprocessor


class ModelTrainer:
    """Handle model training and evaluation."""

    def __init__(
            self,
            test_size: float = 0.3,
            random_state: int = 42,
            cv: int = 5,
            scoring: str = 'accuracy',
            n_jobs: int = -1,
    ):
        """
        Initialize model trainer.

        Parameters
        ----------
        :param test_size: float
            Proportion of dataset to use for testing.
        :param random_state: int
            Random seed.
        :param cv: int
            Number of cross-validation folds.
        :param scoring: str
            Scoring metric
        :param n_jobs: int
            Number of jobs to run in parallel.
        """
        self.results = None
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def prepare_data(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        :param X: np.ndarray
            Feature matrix.
        :param y: np.ndarray
            Target vector.
        :return: tuple
            (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        preprocessor = DataPreprocessor()
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_single_model(
            self,
            model: BaseEstimator,
            X_train: np.ndarray,
            y_train: np.ndarray,
            param_grid: Dict[str, Any],
            model_name: str = None,
    ) -> Dict[str, Any]:
        """
        Train a single model with GridSearchCV.

        :param model: BaseEstimator
            Model for training.
        :param X_train: np.ndarray
            Feature matrix for training model.
        :param y_train: np.ndarray
            Target vector for training model.
        :param param_grid: dict
            Parameter grid for GridSearchCV.
        :param model_name: str
            Name of training model.

        :return: dict
            Dictionary of training results.
        """
        if model_name is None:
            model_name = model.__class__.__name__

        print('Training model:', model_name)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                return_train_score=True,
            )

            grid_search.fit(X_train, y_train)

        results = {
            "Model": model_name,
            "Best parameters": grid_search.best_params_,
            "Best score": grid_search.best_score_,
            "CV results": grid_search.cv_results_,
            "Grid search": grid_search,
        }
        print(f"Best{self.scoring}: {grid_search.best_score_:.4f}"
              f"Best parameters: {grid_search.best_params_}")

        return results

    def train_all_models(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            models_to_train: List[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all models.

        :param X_train: np.ndarray
            Feature matrix for training model.
        :param y_train: np.ndarray
            Target vector for training model.
        :param models_to_train: list, optional
            List of model to train.

        :return: dict
            Dictionary of training results for all models.
        """

        X_train, X_test, y_train, y_test = self.prepare_data(X_train, y_train)

        classifiers = get_all_classifiers()

        if models_to_train:
            classifiers = {
                k: v for k, v in classifiers.items()
                if k in models_to_train
            }

        self.results = {}

        for model_name, (model, param_grid) in classifiers.items():
            try:
                results = self.train_single_model(
                    model=model,
                    param_grid=param_grid,
                    X_train=X_train,
                    y_train=y_train,
                    model_name=model_name,
                )
                test_score = results["Grid search"].score(X_test, y_test)
                results["Test Score"] = test_score

                self.results[model_name] = results
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue

        return self.results

    def get_best_model(self) -> Tuple[str, BaseEstimator]:
        """
        Get best model based on test score.

        :return: tuple
            (model_name, model)

        :raises: ValueError
        """

        if not self.results:
            raise ValueError("No models trained.")

        best_model_name = max(
            self.results.items(),
            key=lambda x: x[1]["Test Score"]
        )[0]

        return best_model_name, self.results[best_model_name]

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table of all models.

        :return: pd.DataFrame
            Comparison table.
        """
        rows = []
        for model_name, results in self.results.items():
            row = {
                "Model": model_name,
                "Test Score": results.get("Test Score", np.nan),
                f"CV {self.scoring.title()}": results["Best score"],
                "Best parameters": str(results["Best parameters"]),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.sort_values(f"CV {self.scoring.title()}", ascending=False)

        return df