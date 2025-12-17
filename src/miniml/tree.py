"""
Decision tree models: Decision Tree Regressor and Classifier.
"""
import numpy as np


# =============================================================================
# Node Classes
# =============================================================================

class Node:
    """
    Base node class for decision trees.
    
    Args:
        feature_index: Index of the feature used for splitting
        threshold: Threshold value for the split
        left: Left child node
        right: Right child node
        value: Leaf value (prediction)
    """
    
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None


# =============================================================================
# Decision Tree Regressor
# =============================================================================

class DecisionTreeRegressor:
    """
    Decision Tree for regression tasks.
    Uses variance reduction as the splitting criterion.
    
    Args:
        min_samples_split: Minimum samples required to split a node
        max_depth: Maximum depth of the tree
    
    Example:
        >>> model = DecisionTreeRegressor(min_samples_split=5, max_depth=10)
        >>> model.fit(X_train, Y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, min_samples_split=2, max_depth=5):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, Y):
        """
        Build the decision tree.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            Y: Target values, shape (n_samples,) or (n_samples, n_outputs)
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.num_targets = Y.shape[1]
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self._build_tree(dataset, depth=0)

    def _build_tree(self, dataset, depth):
        n_samples, n_cols = dataset.shape
        n_features = n_cols - self.num_targets
        Y = dataset[:, n_features:]

        if n_samples >= self.min_samples_split and depth < self.max_depth:
            best_split = self._get_best_split(dataset, n_samples, n_features)

            if best_split["variance_red"] > 1e-7:
                left = self._build_tree(best_split["left"], depth + 1)
                right = self._build_tree(best_split["right"], depth + 1)

                return Node(
                    feature_index=best_split["feature_index"],
                    threshold=best_split["threshold"],
                    left=left,
                    right=right
                )

        leaf_value = np.mean(Y, axis=0)
        return Node(value=leaf_value)

    def _get_best_split(self, dataset, n_samples, n_features):
        best = {"variance_red": -np.inf}

        parent_Y = dataset[:, n_features:]
        parent_variance = self._variance(parent_Y)

        for feature_index in range(n_features):
            feature_vals = dataset[:, feature_index]
            unique_vals = np.unique(feature_vals)

            if len(unique_vals) <= 1:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask = feature_vals <= threshold
                right_mask = feature_vals > threshold

                if not left_mask.any() or not right_mask.any():
                    continue

                left = dataset[left_mask]
                right = dataset[right_mask]

                left_Y = left[:, n_features:]
                right_Y = right[:, n_features:]

                w_l = len(left) / n_samples
                w_r = len(right) / n_samples

                variance_red = parent_variance - (
                    w_l * self._variance(left_Y) +
                    w_r * self._variance(right_Y)
                )

                if variance_red > best["variance_red"]:
                    best = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left": left,
                        "right": right,
                        "variance_red": variance_red
                    }

        return best

    def _variance(self, Y):
        if len(Y) == 0:
            return 0.0
        return np.sum(np.var(Y, axis=0, dtype=np.float64))

    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples, n_outputs)
        """
        predictions = np.array([self._predict_row(x, self.root) for x in X])
        return predictions

    def _predict_row(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)

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
            
        mean = np.mean(Y, axis=0)
        ss_tot = np.sum((Y - mean) ** 2, axis=0)
        ss_res = np.sum((Y - Y_pred) ** 2, axis=0)

        r2_vals = 1 - ss_res / ss_tot
        return np.mean(r2_vals)


# =============================================================================
# Decision Tree Classifier
# =============================================================================

class DecisionTreeClassifier:
    """
    Decision Tree for classification tasks.
    Supports Gini impurity and entropy as splitting criteria.
    
    Args:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        criterion: 'gini' or 'entropy'
    
    Example:
        >>> model = DecisionTreeClassifier(max_depth=10, min_samples_split=3, criterion='gini')
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini"):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        
    def fit(self, X, y):
        """
        Build the decision tree.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Target labels, shape (n_samples,)
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        dataset = np.concatenate((X, y), axis=1)
        self.root = self._build_tree(dataset)
    
    def _build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = X.shape
        
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self._get_best_split(dataset, num_samples, num_features)
            if best_split.get("info_gain", 0) > 0:
                left_subtree = self._build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self._build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(
                    feature_index=best_split["feature_index"],
                    threshold=best_split["threshold"],
                    left=left_subtree,
                    right=right_subtree
                )
        
        leaf_value = self._calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def _get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            sorted_vals = np.sort(np.unique(feature_values))
            
            if len(sorted_vals) <= 1:
                continue
            
            thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
            
            for threshold in thresholds:
                dataset_left, dataset_right = self._split(dataset, feature_index, threshold)
                
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y = dataset[:, -1]
                    left_y = dataset_left[:, -1]
                    right_y = dataset_right[:, -1]
                    curr_info_gain = self._information_gain(y, left_y, right_y)
                    
                    if curr_info_gain > max_info_gain:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "info_gain": curr_info_gain
                        }
                        max_info_gain = curr_info_gain
                        
        return best_split
    
    def _split(self, dataset, feature_index, threshold):
        left_mask = dataset[:, feature_index] <= threshold
        right_mask = dataset[:, feature_index] > threshold
        return dataset[left_mask], dataset[right_mask]
    
    def _information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        
        if self.criterion == "gini":
            gain = self._gini_index(parent) - (
                weight_l * self._gini_index(l_child) + 
                weight_r * self._gini_index(r_child)
            )
        else:
            gain = self._entropy(parent) - (
                weight_l * self._entropy(l_child) + 
                weight_r * self._entropy(r_child)
            )
        return gain
    
    def _entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p = len(y[y == cls]) / len(y)
            entropy += -p * np.log2(p + 1e-10)
        return entropy
    
    def _gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p = len(y[y == cls]) / len(y)
            gini += p ** 2
        return 1 - gini
        
    def _calculate_leaf_value(self, Y):
        Y_int = Y.astype(int)
        counts = np.bincount(Y_int)
        return np.argmax(counts)
    
    def predict(self, X):
        """
        Make class predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predicted class labels, shape (n_samples,)
        """
        return np.array([self._make_prediction(x, self.root) for x in X])
    
    def _make_prediction(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._make_prediction(x, node.left)
        else:
            return self._make_prediction(x, node.right)
        
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Args:
            X: Features, shape (n_samples, n_features)
            y: True labels, shape (n_samples,)
            
        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean((y_pred == y).astype(int))
