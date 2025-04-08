# Curve Prediction with Neural Networks

This project uses a neural network to predict the values of a curve given a set of parameters. The network is trained on a dataset containing curves represented by parameters and their corresponding `(x, y)` pairs.

## Dataset

The dataset consists of text files where each file contains:
1. **Parameters**: A list of `k` parameters (e.g., properties or characteristics of the curve).
2. **Curve Data**: A series of `m` `(x, y)` pairs representing the curve.

The input for both models consists of:
- `k` parameters and `m` `x` values (total size = 54).
The output consists of:
- `m` predicted `y` values corresponding to the input `x` values.

## Models

### 1. Neural Network Model

A simple feed-forward neural network is used for curve prediction:
- **Input Layer**: 54 neurons (`k` parameters + `m` `x` values).
- **Hidden Layers**: 2 layers, each with 128 neurons and ReLU activation.
- **Output Layer**: `m` neurons for the predicted `y` values.

The network is trained using:
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with a learning rate of 0.001.

### 2. Gaussian Process Model

A Gaussian Process (GP) model is used as an alternative approach:
- **Kernel**: A combination of a Constant kernel and a Radial Basis Function (RBF) kernel.
- **Regressor**: `GaussianProcessRegressor` from `sklearn`.

The GP model is trained on individual curves and used to predict the values for the test set.

## Training

Both models are trained using the following setup:
- **Dataset Split**: The dataset is split into training (80%) and testing (20%) sets.
- **Training Epochs**:
  - Neural Network: 1500 epochs.
  - Gaussian Process: Model is trained once on the entire training set.

### Evaluation
- **Test RMSE**: The model's performance is evaluated on the test set using Root Mean Squared Error (RMSE).
- **Visualizations**:
  - Training loss and test RMSE over epochs (Neural Network).
  - Best, worst, and median performing curves with true vs. predicted values.
  - Histogram of RMSE values for the test set.