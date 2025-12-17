"""
Metrics and utility functions for machine learning evaluation.
"""
import numpy as np


def mse(y_true, y_pred, average=True):
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: Ground truth values, shape (n_samples,) or (n_samples, n_outputs)
        y_pred: Predicted values, same shape as y_true
        average: If True, return average across all outputs
        
    Returns:
        MSE value(s)
    """
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)
    
    mse_vals = np.mean((y_true - y_pred) ** 2, axis=0)
    if average:
        return np.mean(mse_vals)
    return mse_vals


def r2_score(y_true, y_pred, average=True):
    """
    Calculate R² (coefficient of determination).
    
    Args:
        y_true: Ground truth values, shape (n_samples,) or (n_samples, n_outputs)
        y_pred: Predicted values, same shape as y_true
        average: If True, return average across all outputs
        
    Returns:
        R² value(s)
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    
    r2_vals = 1 - (ss_res / ss_tot)
    
    if average:
        return np.mean(r2_vals)
    return r2_vals


def accuracy(y_true, y_pred, average=True):
    """
    Calculate classification accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: If True, return average across all outputs
        
    Returns:
        Accuracy value(s)
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    acc = np.mean((y_pred == y_true).astype(int), axis=0)
    if average:
        return np.mean(acc)
    return acc


def cross_entropy(y_true, y_pred):
    """
    Calculate cross entropy loss.
    
    Args:
        y_true: One-hot encoded ground truth, shape (n_samples, n_classes)
        y_pred: Predicted probabilities, shape (n_samples, n_classes)
        
    Returns:
        Cross entropy loss value
    """
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def one_hot(y, num_classes=None):
    """
    One-hot encode labels.
    
    Args:
        y: Integer labels, shape (n_samples,)
        num_classes: Number of classes (auto-detected if None)
        
    Returns:
        One-hot encoded array, shape (n_samples, num_classes)
    """
    y = np.asarray(y).astype(int).reshape(-1)
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
    return np.eye(num_classes)[y]
