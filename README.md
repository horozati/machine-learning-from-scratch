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

which simplifies to:

$$
L(W,B) = \frac{1}{2n} \sum_{i=1}^n \sum_{j=1}^r ( \hat{Y}_{ij} - Y_{ij} )^2
$$

---

#### Backward Propagation

For backward propagation, we calculate the partial derivatives with respect to weights and biases using the chain rule.

Define the error:

$$
E = \hat{Y} - Y
$$

##### Gradient with Respect to Weights

$$
\frac{\partial L}{\partial W} = \frac{1}{n} X^\top E
$$

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

#### Code Example

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

---

### Logistic Regression

#### Notation and Dimensions

Let:
- $n$ be the number of samples
- $m$ be the number of input features
- $X \in \mathbb{R}^{n \times m}$ be the input matrix
- $y \in \{0,1\}^n$ be the binary labels (for binary classification)

---

#### Model Parameters

- Weights: $W \in \mathbb{R}^{m \times 1}$
- Bias: $b \in \mathbb{R}$

---

#### Sigmoid Activation

The sigmoid function maps any real value to the range $(0,1)$:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Properties:
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- $\lim_{z \to \infty} \sigma(z) = 1$
- $\lim_{z \to -\infty} \sigma(z) = 0$

---

#### Forward Propagation

$$
z = XW + b
$$

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-(XW + b)}}
$$

---

#### Loss Function (Binary Cross-Entropy)

$$
L(W, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]
$$

---

#### Backward Propagation

Define the error:

$$
E = \hat{y} - y
$$

##### Gradient with Respect to Weights

$$
\frac{\partial L}{\partial W} = \frac{1}{n} X^\top E
$$

##### Gradient with Respect to Bias

$$
\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} E_i
$$

---

#### Gradient Descent Update

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}
$$

$$
b \leftarrow b - \eta \frac{\partial L}{\partial b}
$$

---

#### Prediction

For binary classification:

$$
\hat{y}_{\text{class}} = \begin{cases}
1 & \text{if } \sigma(XW + b) \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

#### Code Example

```python
import numpy as np
from src.miniml import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train model
model = LogisticRegression(iterations=1000, lr=0.01)
model.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

---

### Neural Networks

#### Notation and Dimensions

Let:
- $L$ be the number of layers
- $n_l$ be the number of neurons in layer $l$
- $a^{[l]} \in \mathbb{R}^{n_l}$ be the activations of layer $l$
- $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$ be the weights of layer $l$
- $b^{[l]} \in \mathbb{R}^{n_l}$ be the biases of layer $l$

---

#### Forward Propagation

For each layer $l$:

$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}
$$

$$
a^{[l]} = g^{[l]}(z^{[l]})
$$

where $g^{[l]}$ is the activation function for layer $l$ (ReLU, Sigmoid, Softmax, etc.)

---

#### Common Activation Functions

##### ReLU (Rectified Linear Unit)

$$
\text{ReLU}(z) = \max(0, z)
$$

$$
\text{ReLU}'(z) = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{otherwise}
\end{cases}
$$

##### Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

##### Softmax

For a vector $z \in \mathbb{R}^k$:

$$
\text{Softmax}(z)_j = \frac{e^{z_j}}{\sum_{i=1}^{k} e^{z_i}}
$$

---

#### Loss Functions

##### Mean Squared Error (Regression)

$$
L = \frac{1}{2n} \sum_{i=1}^{n} \lVert \hat{y}_i - y_i \rVert^2
$$

##### Cross-Entropy (Classification)

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{k} y_{ij} \log(\hat{y}_{ij})
$$

---

#### Backward Propagation

Starting from the output layer, we compute gradients using the chain rule:

##### Output Layer

$$
\delta^{[L]} = \frac{\partial L}{\partial z^{[L]}}
$$

##### Hidden Layers

$$
\delta^{[l]} = (W^{[l+1]})^\top \delta^{[l+1]} \odot g'^{[l]}(z^{[l]})
$$

where $\odot$ denotes element-wise multiplication.

##### Parameter Gradients

$$
\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^\top
$$

$$
\frac{\partial L}{\partial b^{[l]}} = \delta^{[l]}
$$

---

#### Gradient Descent Update

$$
W^{[l]} \leftarrow W^{[l]} - \eta \frac{\partial L}{\partial W^{[l]}}
$$

$$
b^{[l]} \leftarrow b^{[l]} - \eta \frac{\partial L}{\partial b^{[l]}}
$$

---

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

---

### Neural Network Regressor

```python
import numpy as np
from src.miniml import NeuralNetRegressor, Dense, ReLU
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Reshape targets
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Build model
model = NeuralNetRegressor(epochs=500, lr=0.001, layers=[
    Dense(10, 32, "L2"),
    ReLU(),
    Dense(32, 16, "L2"),
    ReLU(),
    Dense(16, 1, "L2")
], verbose=True)

# Train and evaluate
model.fit(X_train, y_train)
from src.miniml.metrics import r2_score
print(f"R² Score: {r2_score(y_test, model.predict(X_test)):.4f}")
```

---

### Decision Tree Classifier

#### Notation and Dimensions

Let:
- $X \in \mathbb{R}^{n \times m}$ be the input data
- $y \in \{0,1,\dots,C-1\}^n$ be the class labels
- $n$ be the number of samples
- $m$ be the number of features
- $C$ be the number of classes

---

#### Dataset at a Node

At a given node $t$, let:
- $S_t$ be the set of samples reaching the node
- $|S_t|$ be the number of samples in the node

For class $c$, the empirical class probability is

$$
p_{t,c} = \frac{1}{|S_t|} \sum_{i \in S_t} \mathbf{1}(y_i = c)
$$

---

#### Impurity Measures

##### Entropy

$$
H(S_t) = -\sum_{c=1}^{C} p_{t,c} \log_2(p_{t,c})
$$

##### Gini Impurity

$$
G(S_t) = 1 - \sum_{c=1}^{C} p_{t,c}^2
$$

---

#### Splitting a Node

A split is defined by:
- feature index $j$
- threshold $s$

The dataset is partitioned as

$$
S_t^{\text{left}} = \left\{ x_i \in S_t \;\middle|\; x_{ij} \le s \right\}
$$

$$
S_t^{\text{right}} = \left\{ x_i \in S_t \;\middle|\; x_{ij} > s \right\}
$$

---

#### Split Quality

##### Information Gain (Entropy)

$$
IG(S_t, j, s) = H(S_t) - \frac{|S_t^{\text{left}}|}{|S_t|} H(S_t^{\text{left}}) - \frac{|S_t^{\text{right}}|}{|S_t|} H(S_t^{\text{right}})
$$

##### Gini Gain

$$
\Delta G(S_t, j, s) = G(S_t) - \frac{|S_t^{\text{left}}|}{|S_t|} G(S_t^{\text{left}}) - \frac{|S_t^{\text{right}}|}{|S_t|} G(S_t^{\text{right}})
$$

---

#### Best Split Selection

The optimal split is chosen as

$$
(j^*, s^*) = \arg\max_{j,s} \Delta G(S_t, j, s)
$$

---

#### Leaf Node Prediction

At a leaf node, the predicted class is

$$
\hat{y} = \arg\max_{c} p_{t,c}
$$

---

#### Stopping Criteria

A node becomes a leaf if any of the following holds:
- all samples belong to the same class
- the maximum depth is reached
- $|S_t| < \text{min\_samples\_split}$
- the impurity gain is zero

---

#### Prediction for a Sample

A test sample $x$ is classified by recursively applying the learned split rules:

$$
\hat{y}(x) = \text{class of the leaf node reached by } x
$$

#### Code Example

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

---

### Decision Tree Regressor

#### Notation and Dimensions

Let:
- $X \in \mathbb{R}^{n \times m}$ be the input data
- $y \in \mathbb{R}^n$ be the continuous target values
- $n$ be the number of samples
- $m$ be the number of features

---

#### Dataset at a Node

At a given node $t$, let:
- $S_t$ be the set of samples reaching the node
- $|S_t|$ be the number of samples in the node

The mean target value is

$$
\bar{y}_t = \frac{1}{|S_t|} \sum_{i \in S_t} y_i
$$

---

#### Impurity Measure (Variance)

The variance (mean squared error) at node $t$ is

$$
\text{Var}(S_t) = \frac{1}{|S_t|} \sum_{i \in S_t} (y_i - \bar{y}_t)^2
$$

---

#### Splitting a Node

A split is defined by:
- feature index $j$
- threshold $s$

The dataset is partitioned as

$$
S_t^{\text{left}} = \left\{ x_i \in S_t \;\middle|\; x_{ij} \le s \right\}
$$

$$
S_t^{\text{right}} = \left\{ x_i \in S_t \;\middle|\; x_{ij} > s \right\}
$$

---

#### Split Quality (Variance Reduction)

$$
\Delta \text{Var}(S_t, j, s) = \text{Var}(S_t) - \frac{|S_t^{\text{left}}|}{|S_t|} \text{Var}(S_t^{\text{left}}) - \frac{|S_t^{\text{right}}|}{|S_t|} \text{Var}(S_t^{\text{right}})
$$

---

#### Best Split Selection

The optimal split maximizes variance reduction:

$$
(j^*, s^*) = \arg\max_{j,s} \Delta \text{Var}(S_t, j, s)
$$

---

#### Leaf Node Prediction

At a leaf node, the predicted value is the mean of all samples:

$$
\hat{y} = \bar{y}_t = \frac{1}{|S_t|} \sum_{i \in S_t} y_i
$$

---

#### Stopping Criteria

A node becomes a leaf if any of the following holds:
- the maximum depth is reached
- $|S_t| < \text{min\_samples\_split}$
- the variance reduction is below a threshold
- all target values are identical

---

#### Prediction for a Sample

A test sample $x$ is predicted by recursively applying the learned split rules:

$$
\hat{y}(x) = \text{mean value of the leaf node reached by } x
$$

#### Code Example

```python
from src.miniml import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train
model = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
model.fit(X_train, y_train)

# Evaluate
from src.miniml.metrics import r2_score
print(f"R² Score: {r2_score(y_test, model.predict(X_test)):.4f}")
```

---

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

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Or run individual test files:

```bash
python tests/test_linear.py
python tests/test_tree.py
```

---

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

---

## License

MIT
