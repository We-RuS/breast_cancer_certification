"""Visualization module."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
from typing import Tuple, Optional
import pandas as pd


class ModelVisualizer:
    """Handle model visualization."""

    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize visualizer.
        :param style: str
            Matplotlib style.
        """
        plt.style.use(style)
        self.colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    def plot_confusion_matrix(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              title: str = "Confusion matrix",
                              normalize: bool = False,
                              ax: Optional[plt.Axes] = None,
                              ) -> plt.Axes:
        """
        Plot confusion matrix.
        :param y_true: np.ndarray
            True labels.
        :param y_pred: np.ndarray
            Predicted labels.
        :param title: str
            Plot title.
        :param normalize: bool
            Whether to normalize the matrix.
        :param ax: plt.Axes, optional
            Axes to plot on.

        :return: plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        sns.heatmap(cm, annot=True, ax=ax, fmt=fmt, cmap="Blues",
                    square=True,cbar_kws={"shrink": .8})
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)

        return ax

    def plot_roc_curve(self, y_true: np.ndarray,
                       y_pred_proba: np.ndarray,
                       model_name: str = "Model",
                       ax: Optional[plt.Axes] = None
                       ) -> Tuple[plt.Axes, float]:
        """
        Plot ROC curve.
        :param y_true:
            True labels.
        :param y_pred_proba:
            Predicted probabilities.
        :param model_name:
            Model name.
        :param ax:
            Axes to plot on.

        :return: tuple (ax, auc_score)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=1,
                label=f"{model_name} (AUC = {roc_auc:.3f})")
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return ax, roc_auc

    def plot_learning_curves(self, model, X: np.ndarray, y: np.ndarray,
                             title: str = "Learning Curves",
                             cv:int = 5, scoring:str = "accuracy",
                             ax: Optional[plt.Axes] = None
                             ) -> plt.Axes:
        """
        Plot learning curve.
        :param model:
        Model to evaluate.
        :param X: np.ndarray
        Features
        :param y: np.ndarray
        Labels
        :param title: str
        Plot title.
        :param cv: int
        Number of cross-validation folds.
        :param scoring: str
        Scoring method.
        :param ax: plt.Axes, optional
        Axes to plot on.

        :return: plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, scoring=scoring,
            train_sizes=np.linspace(.1, 1.0, 10),
            n_jobs=-1
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        ax.fill_between(train_sizes,
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std,
                        alpha=0.1, color=self.colors[0])
        ax.fill_between(train_sizes,
                        test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std,
                        alpha=0.1, color=self.colors[1])

        ax.plot(train_sizes, train_scores_mean, 'o-',
                color=self.colors[0], label='Train score')
        ax.plot(train_sizes, test_scores_mean, 'o-',
                color=self.colors[1], label='Cross-validation score')

        ax.set_xlabel('Training examples', fontsize=12)
        ax.set_ylabel(scoring.title(), fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        return ax

