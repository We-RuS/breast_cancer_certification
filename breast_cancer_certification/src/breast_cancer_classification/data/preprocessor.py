"""Data preprocessing module."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Handle data preprocessing steps"""

    def __init__(self, scale: bool = True, impute: bool = True) -> None:
        """
        Initialize the data preprocessor.

        :param scale: bool
            Whether to scale the data before preprocessing.
        :param impute: bool
            Whether to impute missing values.
        """
        self.scale = scale
        self.impute = impute
        self.scaler = StandardScaler() if scale else None
        self.imputer = SimpleImputer(strategy="mean") if impute else None

    def fit(self, X: np.ndarray) -> "DataPreprocessor":
        """
        Fit preprocessor on data

        :param X: np.ndarray
            Data to fit.
        :return: object
            Fitted preprocessor.
        """
        if self.impute:
            self.imputer.fit(X)

        if self.scale:
            X_to_scale = self.imputer.transform(X)
            self.scaler.fit(X_to_scale)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor
        :param X: np.ndarray
            Data to transform.

        :return: np.ndarray
            Transformed data.
        """
        X_transformed = X.copy()

        if self.impute:
            X_transformed = self.imputer.transform(X_transformed)

        if self.scale:
            X_transformed = self.scaler.transform(X_transformed)

        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit and transform data in one step"""
        return self.fit(X).transform(X)
