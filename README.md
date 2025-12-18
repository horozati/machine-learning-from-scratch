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

Linear Regression (Multi-output)
Notation and Dimensions

Let:

n
n: number of samples

m
m: number of input features

r
r: number of output (label) dimensions

Inputs
X∈Rn×m
X∈R
n×m
X=[x11	x12	⋯	x1m
x21	x22	⋯	x2m
⋮	⋮	⋱	⋮
xn1	xn2	⋯	xnm]
X=
​x11​x21​⋮xn1​​x12​x22​⋮xn2​​⋯⋯⋱⋯​x1m​x2m​⋮xnm​​
​

Targets
Y∈Rn×r
Y∈R
n×r
Y=[y11	y12	⋯	y1r
y21	y22	⋯	y2r
⋮	⋮	⋱	⋮
yn1	yn2	⋯	ynr]
Y=
​y11​y21​⋮yn1​​y12​y22​⋮yn2​​⋯⋯⋱⋯​y1r​y2r​⋮ynr​​
​

Parameters

Weights

W∈Rm×r
W∈R
m×r
W=[w11	w12	⋯	w1r
w21	w22	⋯	w2r
⋮	⋮	⋱	⋮
wm1	wm2	⋯	wmr]
W=
​w11​w21​⋮wm1​​w12​w22​⋮wm2​​⋯⋯⋱⋯​w1r​w2r​⋮wmr​​
​


Bias

B∈R1×r
B∈R
1×r
B=[b1	b2	⋯	br]
B=[
b
1
	​

	​

b
2
	​

	​

⋯
	​

b
r
	​

	​

]
Forward Propagation

The model predicts outputs using a linear transformation:

Y^=XW+B
Y
^
=XW+B

XW∈Rn×r
XW∈R
n×r

Bias 
B
B is broadcast across all samples

Loss Function (Mean Squared Error)

We use mean squared error (MSE) averaged over samples:

Matrix Form
L(W,B)=12n∥Y^−Y∥F2
L(W,B)=
2n
1
	​

∥
Y
^
−Y∥
F
2
	​


where 
∥⋅∥F
∥⋅∥
F
	​

 is the Frobenius norm.

Expanded Form
L(W,B)=12n∑i=1n∑j=1r(∑k=1mXikWkj+Bj−Yij)2
L(W,B)=
2n
1
	​

i=1
∑
n
	​

j=1
∑
r
	​

(
k=1
∑
m
	​

X
ik
	​

W
kj
	​

+B
j
	​

−Y
ij
	​

)
2
Backward Propagation

Define the error matrix:

E=Y^−Y∈Rn×r
E=
Y
^
−Y∈R
n×r
Gradient w.r.t. Weights
∂L∂W=1nX⊤E
∂W
∂L
	​

=
n
1
	​

X
⊤
E

Shape: 
m×r
m×r

Gradient w.r.t. Bias
∂L∂B=1n∑i=1nEi
∂B
∂L
	​

=
n
1
	​

i=1
∑
n
	​

E
i
	​


Equivalent to summing over rows

Shape: 
1×r
1×r

Parameter Update (Gradient Descent)

With learning rate 
η
η:

W←W−η∂L∂W
W←W−η
∂W
∂L
	​

B←B−η∂L∂B
B←B−η
∂B
∂L
	​

Summary (One Iteration)

Forward

Y^=XW+B
Y
^
=XW+B

Compute error

E=Y^−Y
E=
Y
^
−Y

Gradients

∇W=1nX⊤E,∇B=1n∑E
∇
W
	​

=
n
1
	​

X
⊤
E,∇
B
	​

=
n
1
	​

∑E

Update

W,B←W,B−η∇
W,B←W,B−η∇

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
