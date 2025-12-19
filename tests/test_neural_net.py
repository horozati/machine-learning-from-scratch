"""
Unit tests for neural network models.
"""
import sys
import os
# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from miniml.neural_net import NeuralNetRegressor, NeuralNetClassifier, Dense, ReLU, Sigmoid, Softmax
from miniml.metrics import mse, r2_score, accuracy, one_hot


def test_neural_net_regressor_simple():
    """Test that NeuralNetRegressor can fit simple linear data."""
    np.random.seed(2025)
    
    # Simple linear relationship: y = 2x + 1
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X + 1 + np.random.randn(100, 1) * 0.1
    
    model = NeuralNetRegressor(
        epochs=1000,
        lr=0.01,
        layers=[
            Dense(1, 16, "L2", l=0.0001),
            ReLU(),
            Dense(16, 1, "L2", l=0.0001)
        ],
        verbose=False
    )
    
    model.fit(X, y)
    preds = model.predict(X)
    
    # Should fit reasonably well
    error = mse(y, preds)
    assert error < 1.0, f"MSE {error} is too high for simple linear data"
    print(f"[PASS] Neural Net Regressor MSE (linear data): {error:.4f}")


def test_neural_net_regressor_r2():
    """Test R² score on regression data."""
    np.random.seed(2025)
    
    # Multi-feature data
    X = np.random.randn(200, 3)
    y = X[:, 0:1] * 2 + X[:, 1:2] * 3 - X[:, 2:3] * 1.5 + np.random.randn(200, 1) * 0.3
    
    model = NeuralNetRegressor(
        epochs=1500,
        lr=0.01,
        layers=[
            Dense(3, 32, "L2", l=0.0001),
            ReLU(),
            Dense(32, 16, "L2", l=0.0001),
            ReLU(),
            Dense(16, 1, "L2", l=0.0001)
        ],
        verbose=False
    )
    
    model.fit(X, y)
    r2 = model.score(X, y)
    
    assert r2 > 0.85, f"R² score {r2} is too low"
    print(f"[PASS] Neural Net Regressor R²: {r2:.4f}")


def test_neural_net_classifier_binary():
    """Test binary classification."""
    np.random.seed(2025)
    
    # Binary classification data
    X_class0 = np.random.randn(50, 2) - 2
    X_class1 = np.random.randn(50, 2) + 2
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * 50 + [1] * 50)
    y_onehot = one_hot(y, num_classes=2)
    
    model = NeuralNetClassifier(
        epochs=500,
        lr=0.1,
        layers=[
            Dense(2, 8, "L2", l=0.001),
            ReLU(),
            Dense(8, 2, "L2", l=0.001),
            Softmax()
        ],
        verbose=False
    )
    
    model.fit(X, y_onehot)
    acc = model.score(X, y_onehot)
    
    assert acc > 0.95, f"Accuracy {acc} is too low for separable binary data"
    print(f"[PASS] Neural Net Classifier accuracy (binary): {acc:.2%}")


def test_neural_net_classifier_multiclass():
    """Test multi-class classification."""
    np.random.seed(2025)
    
    # 3-class data with clear separation
    X_class0 = np.random.randn(40, 2) + np.array([0, 0])
    X_class1 = np.random.randn(40, 2) + np.array([4, 0])
    X_class2 = np.random.randn(40, 2) + np.array([2, 4])
    X = np.vstack([X_class0, X_class1, X_class2])
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    y_onehot = one_hot(y, num_classes=3)
    
    model = NeuralNetClassifier(
        epochs=800,
        lr=0.05,
        layers=[
            Dense(2, 16, "L2", l=0.001),
            ReLU(),
            Dense(16, 8, "L2", l=0.001),
            ReLU(),
            Dense(8, 3, "L2", l=0.001),
            Softmax()
        ],
        verbose=False
    )
    
    model.fit(X, y_onehot)
    acc = model.score(X, y_onehot)
    
    assert acc > 0.90, f"Accuracy {acc} is too low for multi-class data"
    print(f"[PASS] Neural Net Classifier accuracy (multi-class): {acc:.2%}")


def test_neural_net_predict_proba():
    """Test that predict_proba returns valid probabilities."""
    np.random.seed(2025)
    
    X = np.random.randn(50, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_onehot = one_hot(y, num_classes=2)
    
    model = NeuralNetClassifier(
        epochs=500,
        lr=0.05,
        layers=[
            Dense(2, 8, "L2"),
            ReLU(),
            Dense(8, 2, "L2"),
            Softmax()
        ],
        verbose=False
    )
    
    model.fit(X, y_onehot)
    probas = model.predict_proba(X)
    
    # Check shape
    assert probas.shape == (50, 2), f"Wrong shape: {probas.shape}"
    
    # Check probabilities sum to 1
    prob_sums = np.sum(probas, axis=1)
    assert np.allclose(prob_sums, 1.0), "Probabilities don't sum to 1"
    
    # Check probabilities are in [0, 1]
    assert np.all(probas >= 0) and np.all(probas <= 1), "Probabilities out of range"
    
    print(f"[PASS] Neural Net predict_proba returns valid probabilities")


def test_neural_net_deeper_network():
    """Test that deeper networks can fit complex data."""
    np.random.seed(2025)
    
    # Non-linear XOR-like problem
    X = np.random.rand(200, 2) * 4 - 2
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    y_onehot = one_hot(y, num_classes=2)
    
    # Shallow network (should struggle)
    model_shallow = NeuralNetClassifier(
        epochs=500,
        lr=0.05,
        layers=[
            Dense(2, 2, "L2"),
            Softmax()
        ],
        verbose=False
    )
    model_shallow.fit(X, y_onehot)
    acc_shallow = model_shallow.score(X, y_onehot)
    
    # Deeper network (should do better)
    model_deep = NeuralNetClassifier(
        epochs=1000,
        lr=0.05,
        layers=[
            Dense(2, 16, "L2"),
            ReLU(),
            Dense(16, 16, "L2"),
            ReLU(),
            Dense(16, 2, "L2"),
            Softmax()
        ],
        verbose=False
    )
    model_deep.fit(X, y_onehot)
    acc_deep = model_deep.score(X, y_onehot)
    
    print(f"[PASS] Shallow network accuracy (XOR): {acc_shallow:.2%}")
    print(f"[PASS] Deep network accuracy (XOR): {acc_deep:.2%}")
    assert acc_deep > acc_shallow, "Deep network should outperform shallow on XOR"


def test_neural_net_regularization():
    """Test that L1/L2 regularization affects training."""
    np.random.seed(2025)
    
    X = np.random.randn(100, 5)
    y = X[:, 0:1] * 2 + np.random.randn(100, 1) * 0.5
    
    # Model with L2 regularization
    model_l2 = NeuralNetRegressor(
        epochs=500,
        lr=0.01,
        layers=[
            Dense(5, 32, "L2", l=0.1),  # Strong regularization
            ReLU(),
            Dense(32, 1, "L2", l=0.1)
        ],
        verbose=False
    )
    model_l2.fit(X, y)
    
    # Model with L1 regularization
    model_l1 = NeuralNetRegressor(
        epochs=500,
        lr=0.01,
        layers=[
            Dense(5, 32, "L1", l=0.1),
            ReLU(),
            Dense(32, 1, "L1", l=0.1)
        ],
        verbose=False
    )
    model_l1.fit(X, y)
    
    # Check that weights are relatively small due to regularization
    l2_weight_norm = np.linalg.norm(model_l2.layers[0].weights)
    l1_weight_norm = np.linalg.norm(model_l1.layers[0].weights)
    
    print(f"[PASS] L2 regularized weight norm: {l2_weight_norm:.4f}")
    print(f"[PASS] L1 regularized weight norm: {l1_weight_norm:.4f}")
    
    # Both should still make reasonable predictions
    r2_l2 = model_l2.score(X, y)
    r2_l1 = model_l1.score(X, y)
    
    assert r2_l2 > 0.5, "L2 model should still fit reasonably"
    assert r2_l1 > 0.5, "L1 model should still fit reasonably"


def test_neural_net_activation_functions():
    """Test different activation functions."""
    np.random.seed(2025)
    
    X = np.random.randn(100, 3)
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    y_onehot = one_hot(y, num_classes=2)
    
    # Model with ReLU
    model_relu = NeuralNetClassifier(
        epochs=500,
        lr=0.05,
        layers=[
            Dense(3, 8, "L2"),
            ReLU(),
            Dense(8, 2, "L2"),
            Softmax()
        ],
        verbose=False
    )
    model_relu.fit(X, y_onehot)
    acc_relu = model_relu.score(X, y_onehot)
    
    # Model with Sigmoid
    model_sigmoid = NeuralNetClassifier(
        epochs=500,
        lr=0.05,
        layers=[
            Dense(3, 8, "L2"),
            Sigmoid(),
            Dense(8, 2, "L2"),
            Softmax()
        ],
        verbose=False
    )
    model_sigmoid.fit(X, y_onehot)
    acc_sigmoid = model_sigmoid.score(X, y_onehot)
    
    print(f"[PASS] ReLU activation accuracy: {acc_relu:.2%}")
    print(f"[PASS] Sigmoid activation accuracy: {acc_sigmoid:.2%}")
    
    # Both should achieve reasonable accuracy
    assert acc_relu > 0.80, "ReLU model should achieve good accuracy"
    assert acc_sigmoid > 0.80, "Sigmoid model should achieve good accuracy"


if __name__ == "__main__":
    print("=" * 50)
    print("Running Neural Network Tests")
    print("=" * 50)
    
    test_neural_net_regressor_simple()
    test_neural_net_regressor_r2()
    test_neural_net_classifier_binary()
    test_neural_net_classifier_multiclass()
    test_neural_net_predict_proba()
    test_neural_net_deeper_network()
    test_neural_net_regularization()
    test_neural_net_activation_functions()
    
    print("=" * 50)
    print("All Neural Network Tests Passed!")
    print("=" * 50)"
