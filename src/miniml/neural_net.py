"""
Neural network components: layers, activations, and network classes.
"""
import numpy as np
from .metrics import cross_entropy


# =============================================================================
# Layer Classes
# =============================================================================

class Dense:
    """
    Fully connected (dense) layer.
    
    Args:
        n_inputs: Number of input features
        n_outputs: Number of output neurons
        regularization: 'L1', 'L2', or None
        l: Regularization strength
    """
    
    def __init__(self, n_inputs, n_outputs, regularization=None, l=0.001):
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.01
        self.biases = np.zeros((1, n_outputs))
        self.regularization = regularization
        self.l = l

    def forward(self, X):
        """Forward pass: Z = X @ W + b"""
        self.X = X
        # (n_samples, n_inputs) @ (n_inputs, n_outputs) = (n_samples, n_outputs)
        self.Z = X @ self.weights + self.biases
        return self.Z

    def backward(self, grad, lr):
        """Backward pass with weight update."""
        n_samples = self.X.shape[0]
        
        # dW: (n_inputs, n_samples) @ (n_samples, n_outputs) = (n_inputs, n_outputs)
        dW = (self.X.T @ grad) / n_samples
        dB = np.sum(grad, axis=0, keepdims=True) / n_samples
        
        # dX: (n_samples, n_outputs) @ (n_outputs, n_inputs) = (n_samples, n_inputs)
        dX = grad @ self.weights.T
        
        if self.regularization == "L2":
            dW += (self.l / n_samples) * self.weights
        elif self.regularization == "L1":
            dW += (self.l / n_samples) * np.sign(self.weights)
        
        self.weights -= lr * dW
        self.biases -= lr * dB
        
        return dX


# =============================================================================
# Activation Classes
# =============================================================================

class ReLU:
    """Rectified Linear Unit activation."""
    
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, grad, lr):
        return grad * (self.Z > 0)


class Sigmoid:
    """Sigmoid activation."""
    
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        return self.A

    def backward(self, grad, lr):
        return grad * self.A * (1 - self.A)


class Softmax:
    """Softmax activation for multi-class classification."""
    
    def forward(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.output = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.output

    def backward(self, grad, lr):
        # Simplified gradient when combined with cross-entropy loss
        return grad


# =============================================================================
# Neural Network Classes
# =============================================================================

class NeuralNetRegressor:
    """
    Neural network for regression tasks.
    
    Args:
        epochs: Number of training epochs
        lr: Learning rate
        layers: List of layer objects (Dense, ReLU, etc.)
        verbose: Whether to print training progress
    
    Example:
        >>> model = NeuralNetRegressor(1000, 0.01, [
        ...     Dense(4, 64, "L2", l=0.0001),
        ...     ReLU(),
        ...     Dense(64, 1, "L2", l=0.0001)
        ... ], verbose=True)
        >>> model.fit(X_train, Y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, epochs, lr, layers, verbose=False):
        self.verbose = verbose
        self.epochs = epochs
        self.lr = lr
        self.layers = layers

    def fit(self, X, Y):
        """
        Train the neural network.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            Y: Target values, shape (n_samples,) or (n_samples, n_outputs)
        """
        n_samples = X.shape[0]
        
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        for i in range(1, self.epochs + 1):
            # Forward pass
            Y_hat = X
            for layer in self.layers:
                Y_hat = layer.forward(Y_hat)
           
            # MSE loss
            loss = np.sum((Y_hat - Y) ** 2) / (2 * n_samples)
            
            # Add regularization to loss
            for layer in self.layers:
                if isinstance(layer, Dense):
                    if layer.regularization == "L2":
                        loss += (layer.l / (2 * n_samples)) * np.sum(layer.weights ** 2)
                    elif layer.regularization == "L1":
                        loss += (layer.l / n_samples) * np.sum(np.abs(layer.weights))
            
            # Backward pass
            loss_grad = (Y_hat - Y)
            for layer in reversed(self.layers):
                loss_grad = layer.backward(loss_grad, self.lr)

            if self.verbose and i % max(1, self.epochs // 10) == 0:
                print(f"Epoch {i}: Loss = {loss:.4f}")

    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples, n_outputs)
        """
        Y_hat = X
        for layer in self.layers:
            Y_hat = layer.forward(Y_hat)
        return Y_hat
    
    def score(self, X, Y):
        """
        Calculate R² score.
        
        Args:
            X: Features, shape (n_samples, n_features)
            Y: True values, shape (n_samples,) or (n_samples, n_outputs)
            
        Returns:
            R² score
        """
        Y_pred = self.predict(X)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y_pred.ndim == 1:
            Y_pred = Y_pred.reshape(-1, 1)
            
        ss_res = np.sum((Y - Y_pred) ** 2, axis=0)
        ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
        r2_scores = 1 - (ss_res / ss_tot)
        
        return np.mean(r2_scores)


class NeuralNetClassifier:
    """
    Neural network for classification tasks.
    
    Args:
        epochs: Number of training epochs
        lr: Learning rate
        layers: List of layer objects (should end with Softmax for multi-class)
        verbose: Whether to print training progress
    
    Example:
        >>> model = NeuralNetClassifier(1000, 0.01, [
        ...     Dense(4, 16, "L2"),
        ...     ReLU(),
        ...     Dense(16, 3, "L2"),
        ...     Softmax()
        ... ], verbose=True)
        >>> model.fit(X_train, Y_train_onehot)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, epochs, lr, layers, verbose=False):
        self.verbose = verbose
        self.epochs = epochs
        self.lr = lr
        self.layers = layers

    def fit(self, X, Y):
        """
        Train the neural network.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            Y: One-hot encoded targets, shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        
        for i in range(1, self.epochs + 1):
            # Forward pass
            Y_hat = X
            for layer in self.layers:
                Y_hat = layer.forward(Y_hat)
            
            assert Y_hat.shape == Y.shape, f"Shape mismatch: Predicted {Y_hat.shape} vs Target {Y.shape}"
            
            # Cross-entropy loss
            loss = cross_entropy(Y, Y_hat)
            
            # Add regularization
            for layer in self.layers:
                if isinstance(layer, Dense):
                    if layer.regularization == "L2":
                        loss += (layer.l / (2 * n_samples)) * np.sum(layer.weights ** 2)
                    elif layer.regularization == "L1":
                        loss += (layer.l / n_samples) * np.sum(np.abs(layer.weights))
            
            # Backward pass (gradient for softmax + cross-entropy)
            loss_grad = Y_hat - Y
            
            for layer in reversed(self.layers):
                loss_grad = layer.backward(loss_grad, self.lr)
            
            if self.verbose and i % max(1, self.epochs // 10) == 0:
                print(f"Epoch {i}: Loss = {loss:.4f}")
        
    def predict(self, X):
        """
        Make class predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predicted class indices, shape (n_samples,)
        """
        Y_hat = X
        for layer in self.layers:
            Y_hat = layer.forward(Y_hat)
        return np.argmax(Y_hat, axis=1)
    
    def predict_proba(self, X):
        """
        Get class probabilities.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        Y_hat = X
        for layer in self.layers:
            Y_hat = layer.forward(Y_hat)
        return Y_hat
    
    def score(self, X, Y):
        """
        Calculate accuracy score.
        
        Args:
            X: Features, shape (n_samples, n_features)
            Y: One-hot encoded true labels, shape (n_samples, n_classes)
            
        Returns:
            Accuracy score
        """
        Y_hat = self.predict(X)
        Y_true = np.argmax(Y, axis=1)
        return np.mean(Y_hat == Y_true)
