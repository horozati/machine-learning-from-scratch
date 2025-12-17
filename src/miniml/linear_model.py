"""
Linear models: Linear Regression and Logistic Regression.
"""
import numpy as np


class LinearRegression:
    """
    Linear Regression with optional L1/L2 regularization.
    
    Args:
        iterations: Number of gradient descent iterations
        lr: Learning rate
        regularization: 'L1', 'L2', or None
        l: Regularization strength
    """
    
    def __init__(self, iterations=1000, lr=0.1, regularization=None, l=0.001):
        self.iters = iterations
        self.lr = lr
        self.regularization = regularization
        self.l = l
    
    def fit(self, X, Y):
        """
        Fit the model to training data.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            Y: Target values, shape (n_samples,) or (n_samples, n_outputs)
        """
        n_samples, n_features = X.shape
        
        # Handle both 1D and 2D Y
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        n_outputs = Y.shape[1]
        
        self.weights = np.random.randn(n_features, n_outputs) * 0.01
        self.biases = np.zeros((1, n_outputs))
        
        for i in range(self.iters):
            # Forward pass: (n_samples, n_features) @ (n_features, n_outputs) = (n_samples, n_outputs)
            Y_hat = X @ self.weights + self.biases
            
            # Loss calculation
            loss = np.sum((Y_hat - Y) ** 2) / (2 * n_samples)
            
            if self.regularization == "L1":
                loss += (self.l / n_samples) * np.sum(np.abs(self.weights))
            elif self.regularization == "L2":
                loss += (self.l / (2 * n_samples)) * np.sum(self.weights ** 2)
            
            if i % max(1, self.iters // 10) == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
            
            # Gradients
            grad = (Y_hat - Y) / n_samples
            
            # dW: (n_features, n_samples) @ (n_samples, n_outputs) = (n_features, n_outputs)
            dW = X.T @ grad
            db = np.sum(grad, axis=0, keepdims=True)
            
            if self.regularization == "L1":
                dW += (self.l / n_samples) * np.sign(self.weights)
            elif self.regularization == "L2":
                dW += (self.l / n_samples) * self.weights
            
            self.weights -= self.lr * dW
            self.biases -= self.lr * db
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples, n_outputs)
        """
        return X @ self.weights + self.biases
    
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


class LogisticRegression:
    """
    Logistic Regression with optional L1/L2 regularization.
    Supports multi-label classification.
    
    Args:
        iterations: Number of gradient descent iterations
        lr: Learning rate
        regularization: 'L1', 'L2', or None
        l: Regularization strength
    """
    
    def __init__(self, iterations=2000, lr=0.01, regularization=None, l=0.001):
        self.iters = iterations
        self.lr = lr
        self.regularization = regularization
        self.l = l

    def _sigmoid(self, Z):
        """Numerically stable sigmoid function."""
        return np.clip(1 / (1 + np.exp(-np.clip(Z, -500, 500))), 1e-7, 1 - 1e-7)

    def fit(self, X, Y):
        """
        Fit the model to training data.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            Y: Binary target values, shape (n_samples,) or (n_samples, n_outputs)
        """
        n_samples, n_features = X.shape
        
        # Handle both 1D and 2D Y
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        n_outputs = Y.shape[1]
        
        self.weights = np.random.randn(n_features, n_outputs) * 0.01
        self.bias = np.zeros((1, n_outputs))

        for i in range(self.iters):
            # Forward pass
            Z = X @ self.weights + self.bias
            Y_hat = self._sigmoid(Z)

            # Binary cross-entropy loss
            loss = -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / n_samples

            if self.regularization == "L1":
                loss += (self.l / n_samples) * np.sum(np.abs(self.weights))
            elif self.regularization == "L2":
                loss += (self.l / (2 * n_samples)) * np.sum(self.weights ** 2)

            if i % max(1, self.iters // 10) == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

            # Gradients
            grad = (Y_hat - Y) / n_samples
            
            dW = X.T @ grad
            db = np.sum(grad, axis=0, keepdims=True)

            if self.regularization == "L1":
                dW += (self.l / n_samples) * np.sign(self.weights)
            elif self.regularization == "L2":
                dW += (self.l / n_samples) * self.weights

            self.weights -= self.lr * dW
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Make binary predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Binary predictions, shape (n_samples, n_outputs)
        """
        return (self._sigmoid(X @ self.weights + self.bias) >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Probabilities, shape (n_samples, n_outputs)
        """
        return self._sigmoid(X @ self.weights + self.bias)
    
    def score(self, X, Y):
        """
        Calculate accuracy score.
        
        Args:
            X: Features, shape (n_samples, n_features)
            Y: True labels, shape (n_samples,) or (n_samples, n_outputs)
            
        Returns:
            Accuracy score
        """
        Y_pred = self.predict(X)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y_pred.ndim == 1:
            Y_pred = Y_pred.reshape(-1, 1)
            
        acc = np.mean((Y_pred == Y).astype(int), axis=0)
        return np.mean(acc)
