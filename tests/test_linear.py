"""
Unit tests for linear models.
"""
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from miniml.linear_model import LinearRegression, LogisticRegression
np.random.seed(2025)

def test_linear_fit():
    """Test that LinearRegression can fit a simple y = 2x + 1 pattern."""
    # Create dummy data (y = 2x + 1)
    X = np.array([[1], [2], [3], [4]])
    y = np.array([[3], [5], [7], [9]])  # Shape (4, 1)
    
    # Initialize and fit
    model = LinearRegression(iterations=1000, lr=0.1)
    model.fit(X, y)
    
    # Predict
    pred = model.predict(np.array([[5]]))  # Should be close to 11
    
    # Assert
    assert np.abs(pred[0, 0] - 11) < 0.5, f"Prediction {pred[0, 0]} is too far from 11"
    print(f"[PASS] Linear Regression prediction: {pred[0, 0]:.2f} (expected ~11)")


def test_linear_r2():
    """Test that LinearRegression achieves high R2 on simple data."""
    
    # Generate data: y = 3x + noise
    X = np.random.randn(100, 1) * 2
    y = 3 * X + np.random.randn(100, 1) * 0.1
    
    # Train
    model = LinearRegression(iterations=500, lr=0.1)
    model.fit(X, y)
    
    # Check R2
    r2 = model.score(X, y)
    assert r2 > 0.99, f"R2 score {r2} is too low"
    print(f"[PASS] Linear Regression R2: {r2:.4f}")


def test_linear_regularization():
    """Test that L2 regularization reduces weight magnitude."""
    np.random.seed(42)
    
    X = np.random.randn(50, 5)
    y = X @ np.array([[1], [2], [3], [4], [5]]) + np.random.randn(50, 1) * 0.1
    
    # Without regularization
    model_no_reg = LinearRegression(iterations=500, lr=0.01)
    model_no_reg.fit(X, y)
    
    # With L2 regularization
    model_l2 = LinearRegression(iterations=500, lr=0.01, regularization="L2", l=1.0)
    model_l2.fit(X, y)
    
    # L2 should have smaller weight norm
    norm_no_reg = np.linalg.norm(model_no_reg.weights)
    norm_l2 = np.linalg.norm(model_l2.weights)
    
    assert norm_l2 < norm_no_reg, f"L2 weight norm {norm_l2} should be smaller than {norm_no_reg}"
    print(f"[PASS] L2 regularization reduces weight norm: {norm_no_reg:.2f} -> {norm_l2:.2f}")


def test_logistic_binary():
    """Test LogisticRegression on binary classification."""
    
    # Create separable data
    X_class0 = np.random.randn(50, 2) - 2
    X_class1 = np.random.randn(50, 2) + 2
    X = np.vstack([X_class0, X_class1])
    y = np.array([[0]] * 50 + [[1]] * 50)
    
    # Train
    model = LogisticRegression(iterations=1000, lr=0.1)
    model.fit(X, y)
    
    # Check accuracy
    acc = model.score(X, y)
    assert acc > 0.95, f"Accuracy {acc} is too low for separable data"
    print(f"[PASS] Logistic Regression accuracy: {acc:.2%}")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Linear Model Tests")
    print("=" * 50)
    
    test_linear_fit()
    test_linear_r2()
    test_linear_regularization()
    test_logistic_binary()
    
    print("=" * 50)
    print("All Linear Model Tests Passed!")
    print("=" * 50)

