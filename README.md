# MiniML

A from-scratch machine learning library built with NumPy.

## Overview

This library implements core ML algorithms from the ground up, including:

- **Linear Models**: Linear Regression, Logistic Regression (with L1/L2 regularization)
- **Neural Networks**: Fully-connected networks for regression and classification
- **Decision Trees**: CART-style trees for regression and classification

## Installation

```bash
# Clone the repository
git clone https://github.com/horozati/machine-learning-from-scratch.git
cd machine-learning-from-scratch

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Models

### Linear Regression

#### Notation and Dimensions

Let:
- $n$ be the number of samples  
- $m$ be the number of input features  
- $r$ be the number of output dimensions  

---

#### Input Matrix

$$
X \in \mathbb{R}^{n \times m}
$$

$$
X =
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1m} \\
x_{21} & x_{22} & \cdots & x_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{nm}
\end{bmatrix}
$$

---

#### Target Matrix

$$
Y \in \mathbb{R}^{n \times r}
$$

$$
Y =
\begin{bmatrix}
y_{11} & y_{12} & \cdots & y_{1r} \\
y_{21} & y_{22} & \cdots & y_{2r} \\
\vdots & \vdots & \ddots & \vdots \\
y_{n1} & y_{n2} & \cdots & y_{nr}
\end{bmatrix}
$$

---

#### Model Parameters

##### Weights

$$
W \in \mathbb{R}^{m \times r}
$$

$$
W =
\begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1r} \\
w_{21} & w_{22} & \cdots & w_{2r} \\
\vdots & \vdots & \ddots & \vdots \\
w_{m1} & w_{m2} & \cdots & w_{mr}
\end{bmatrix}
$$

##### Biases

$$
B \in \mathbb{R}^{1 \times r}
$$

$$
B =
\begin{bmatrix}
b_1 & b_2 & \cdots & b_r
\end{bmatrix}
$$

---

#### Forward Propagation

The predicted outputs are computed as:

$$
\hat{Y} = XW + B
$$

---

#### Loss Function (Mean Squared Error)

The loss is defined as the average squared error over all samples:

$$
L(W,B) = \frac{1}{2n} \lVert \hat{Y} - Y \rVert_F^2
$$

##### Expanded Form

$$
L(W,B) =
\frac{1}{2n}
\sum_{i=1}^{n}
\sum_{j=1}^{r}
\left(
\sum_{k=1}^{m} X_{ik} W_{kj} + B_j - Y_{ij}
\right)^2
$$

---

#### Backward Propagation

Define the error matrix:

$$
E = \hat{Y} - Y
$$

---

##### Gradient with Respect to Weights

$$
\frac{\partial L}{\partial W} = \frac{1}{n} X^\top E
$$

---

##### Gradient with Respect to Bias

$$
\frac{\partial L}{\partial B} = \frac{1}{n} \sum_{i=1}^{n} E_i
$$

---

#### Gradient Descent Update

Using learning rate $\eta$:

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}
$$

$$
B \leftarrow B - \eta \frac{\partial L}{\partial B}
$$

```python
import numpy as np
from src.miniml import LinearRegression

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([[2], [4], [6], [8], [10]])  # y = 2x

# Train model
model = LinearRegression(iterations=1000, lr=0.01)
model.fit(X, y)

# Predict
predictions = model.predict(np.array([[6], [7]]))
print(predictions)  # Should be close to [[12], [14]]
```

### Neural Network Classifier

```python
import numpy as np
from src.miniml import NeuralNetClassifier, Dense, ReLU, Softmax, one_hot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and prepare data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# One-hot encode labels
y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# Build model
model = NeuralNetClassifier(epochs=500, lr=0.01, layers=[
    Dense(4, 16, "L2"),
    ReLU(),
    Dense(16, 3, "L2"),
    Softmax()
], verbose=True)

# Train and evaluate
model.fit(X_train, y_train_oh)
print(f"Accuracy: {model.score(X_test, y_test_oh):.2%}")
```

### Decision Tree

```python
from src.miniml import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train
model = DecisionTreeClassifier(max_depth=5, min_samples_split=3, criterion="gini")
model.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

## API Reference

### Linear Models

| Class | Description |
|-------|-------------|
| `LinearRegression` | Linear regression with optional L1/L2 regularization |
| `LogisticRegression` | Binary/multi-label classification with sigmoid activation |

### Neural Networks

| Class | Description |
|-------|-------------|
| `Dense` | Fully connected layer |
| `ReLU` | ReLU activation |
| `Sigmoid` | Sigmoid activation |
| `Softmax` | Softmax activation for multi-class |
| `NeuralNetRegressor` | Neural network for regression |
| `NeuralNetClassifier` | Neural network for classification |

### Decision Trees

| Class | Description |
|-------|-------------|
| `DecisionTreeRegressor` | Regression tree using variance reduction |
| `DecisionTreeClassifier` | Classification tree using Gini/Entropy |

### Metrics

| Function | Description |
|----------|-------------|
| `mse(y_true, y_pred)` | Mean Squared Error |
| `r2_score(y_true, y_pred)` | R² coefficient |
| `accuracy(y_true, y_pred)` | Classification accuracy |
| `cross_entropy(y_true, y_pred)` | Cross-entropy loss |
| `one_hot(y)` | One-hot encoding |

## Running Tests

```bash
python -m pytest tests/ -v
```

Or run individual test files:

```bash
python tests/test_linear.py
python tests/test_tree.py
```

## Project Structure

```
miniml/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   └── miniml/
│       ├── __init__.py
│       ├── linear_model.py
│       ├── neural_net.py
│       ├── tree.py
│       └── metrics.py
└── tests/
    ├── __init__.py
    ├── test_linear.py
    └── test_tree.py
```

## License

MIT
