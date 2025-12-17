"""
MiniML: A from-scratch machine learning library.

This library provides implementations of common ML algorithms including:
- Linear Regression and Logistic Regression
- Neural Networks (Regressor and Classifier)
- Decision Trees (Regressor and Classifier)

Example:
    >>> from miniml import LinearRegression
    >>> model = LinearRegression(iterations=1000, lr=0.01)
    >>> model.fit(X_train, Y_train)
    >>> predictions = model.predict(X_test)
"""

from .linear_model import LinearRegression, LogisticRegression
from .neural_net import (
    NeuralNetRegressor,
    NeuralNetClassifier,
    Dense,
    ReLU,
    Sigmoid,
    Softmax
)
from .tree import DecisionTreeRegressor, DecisionTreeClassifier
from .metrics import mse, r2_score, accuracy, cross_entropy, one_hot

__version__ = "0.1.0"

__all__ = [
    # Linear models
    "LinearRegression",
    "LogisticRegression",
    # Neural networks
    "NeuralNetRegressor",
    "NeuralNetClassifier",
    "Dense",
    "ReLU",
    "Sigmoid",
    "Softmax",
    # Decision trees
    "DecisionTreeRegressor",
    "DecisionTreeClassifier",
    # Metrics
    "mse",
    "r2_score",
    "accuracy",
    "cross_entropy",
    "one_hot",
]
