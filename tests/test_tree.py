"""
Unit tests for decision tree models.
"""
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from miniml.tree import DecisionTreeRegressor, DecisionTreeClassifier


def test_tree_regressor_fit():
    """Test that DecisionTreeRegressor can fit simple data."""
    np.random.seed(2025)
    
    # Create simple pattern
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    y = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])  # y = x
    
    model = DecisionTreeRegressor(min_samples_split=2, max_depth=5)
    model.fit(X, y)
    
    # Predict on training data
    preds = model.predict(X)
    
    # Should be able to fit training data well
    mse = np.mean((preds - y) ** 2)
    assert mse < 1.0, f"MSE {mse} is too high"
    print(f"[PASS] Decision Tree Regressor MSE: {mse:.4f}")


def test_tree_regressor_r2():
    """Test R2 score on regression data."""
    np.random.seed(2025)
    
    # More complex data
    X = np.random.randn(100, 3)
    y = X[:, 0:1] * 2 + X[:, 1:2] * 3 + np.random.randn(100, 1) * 0.5
    
    model = DecisionTreeRegressor(min_samples_split=5, max_depth=8)
    model.fit(X, y)
    
    r2 = model.score(X, y)
    assert r2 > 0.8, f"R2 score {r2} is too low"
    print(f"[PASS] Decision Tree Regressor R2: {r2:.4f}")


def test_tree_classifier_fit():
    """Test that DecisionTreeClassifier can classify simple data."""
    np.random.seed(2025)
    
    # Create separable data
    X_class0 = np.random.randn(30, 2) - 1.5
    X_class1 = np.random.randn(30, 2) + 1.5
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * 30 + [1] * 30)
    
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=2, criterion="gini")
    model.fit(X, y)
    
    # Check training accuracy
    acc = model.score(X, y)
    assert acc > 0.9, f"Accuracy {acc} is too low for separable data"
    print(f"[PASS] Decision Tree Classifier accuracy (Gini): {acc:.2%}")


def test_tree_classifier_entropy():
    """Test classifier with entropy criterion."""
    np.random.seed(2025)
    
    # Multi-class data
    X_class0 = np.random.randn(20, 2) + np.array([0, 0])
    X_class1 = np.random.randn(20, 2) + np.array([3, 0])
    X_class2 = np.random.randn(20, 2) + np.array([1.5, 3])
    X = np.vstack([X_class0, X_class1, X_class2])
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    
    model = DecisionTreeClassifier(max_depth=10, min_samples_split=2, criterion="entropy")
    model.fit(X, y)
    
    acc = model.score(X, y)
    assert acc > 0.9, f"Accuracy {acc} is too low"
    print(f"[PASS] Decision Tree Classifier accuracy (Entropy): {acc:.2%}")


def test_tree_depth_limit():
    """Test that max_depth limits tree complexity."""
    np.random.seed(2025)
    
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    
    # Shallow tree
    model_shallow = DecisionTreeClassifier(max_depth=2)
    model_shallow.fit(X, y)
    
    # Deep tree
    model_deep = DecisionTreeClassifier(max_depth=10)
    model_deep.fit(X, y)
    
    # Deep tree should have higher training accuracy
    acc_shallow = model_shallow.score(X, y)
    acc_deep = model_deep.score(X, y)
    
    print(f"[PASS] Shallow tree (depth=2) accuracy: {acc_shallow:.2%}")
    print(f"[PASS] Deep tree (depth=10) accuracy: {acc_deep:.2%}")
    assert acc_deep >= acc_shallow, "Deep tree should be at least as accurate as shallow"


if __name__ == "__main__":
    print("=" * 50)
    print("Running Decision Tree Tests")
    print("=" * 50)
    
    test_tree_regressor_fit()
    test_tree_regressor_r2()
    test_tree_classifier_fit()
    test_tree_classifier_entropy()
    test_tree_depth_limit()
    
    print("=" * 50)
    print("All Decision Tree Tests Passed!")
    print("=" * 50)

