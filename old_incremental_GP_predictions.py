import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(34)

# Folder containing curve files
data_folder = "curves"

# Load dataset
def load_dataset(folder):
    X, Y = [], []
    
    print("Loading data from folder:", folder)
    
    for file in os.listdir(folder):
        if file.startswith("curve_") and file.endswith(".txt"):
            filepath = os.path.join(folder, file)
            
            with open(filepath, "r") as f:
                lines = f.readlines()
            
            # Extract parameters
            params = eval(lines[0].split(":")[1].strip())  # List of 4 params
            
            # Extract points
            points = [tuple(map(float, line.strip().split(", "))) for line in lines[2:]]
            x_vals, y_vals = zip(*points)  # Separate x and y values
            
            # Convert to numpy arrays
            X.append(params + list(x_vals))  # 4 params + 50 x-values (input size = 54)
            Y.append(list(y_vals))           # 50 y-values (output size = 50)
    
    print(f"Data loaded: {len(X)} samples")
    return np.array(X), np.array(Y)

# Load data
print("Loading dataset...")
X, Y = load_dataset(data_folder)
print(f"Dataset loaded! Total samples: {len(X)}")

# Split into training (80%) and testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Define Gaussian Process model with RBF kernel
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))  # Constant kernel * RBF kernel
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Track RMSE during incremental learning
def track_gp_learning(X_train, Y_train, X_test, Y_test, gp_model, num_steps=10):
    rmse_history = []
    for step in range(1, num_steps + 1):
        # Incrementally select a subset of the training data
        subset_size = int(len(X_train) * step / num_steps)
        X_train_subset = X_train[:subset_size]
        Y_train_subset = Y_train[:subset_size]
        
        # Train GP on the current subset
        print(f"Training GP with {subset_size} training samples...")
        gp_model.fit(X_train_subset, Y_train_subset)
        
        # Make predictions on the test set
        Y_pred, _ = gp_model.predict(X_test, return_std=True)
        
        # Calculate RMSE
        test_mse = np.mean((Y_pred - Y_test) ** 2)
        test_rmse = np.sqrt(test_mse)
        rmse_history.append(test_rmse)
        
        print(f"Step {step}/{num_steps}, Test RMSE: {test_rmse:.6f}")
    
    return rmse_history

# Track the learning process
print("Tracking GP learning progress...")
rmse_history = track_gp_learning(X_train, Y_train, X_test, Y_test, gp_model, num_steps=20)

# Plot the RMSE history
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(rmse_history) + 1), rmse_history, label="Test RMSE")
plt.xlabel("Training Progress (Steps)")
plt.ylabel("RMSE")
plt.title("Test RMSE vs Training Progress (Incremental Learning with GP)")
plt.grid(True)
plt.legend()
plt.show()

# ====== PLOT BEST, WORST, AND MEDIAN RMSE CURVES ======
def plot_best_worst_median_curves(Y_pred, Y_test):
    """Plot the best, worst, and median RMSE curves, with true vs predicted values and RMSE."""

    # Calculate RMSE for each test sample
    test_rmse_values = []
    print("Calculating RMSE for each test sample...")
    for i in range(len(X_test)):
        true_y = Y_test[i]
        predicted_y = Y_pred[i]
        rmse = np.sqrt(np.mean((true_y - predicted_y) ** 2))
        test_rmse_values.append(rmse)
    
    print(f"RMSE values calculated for {len(X_test)} test samples.")

    # Sort the RMSE values and find the best, worst, and median
    rmse_sorted_indices = np.argsort(test_rmse_values)
    
    # Best RMSE (smallest RMSE)
    best_idx = rmse_sorted_indices[0]
    # Worst RMSE (largest RMSE)
    worst_idx = rmse_sorted_indices[-1]
    # Median RMSE
    median_rmse_value = np.median(test_rmse_values)
    median_idx = min(rmse_sorted_indices, key=lambda idx: abs(test_rmse_values[idx] - median_rmse_value))

    # Indices of the best, worst, and median curves
    indices_to_plot = [best_idx, worst_idx, median_idx]

    # Plot the best, worst, and median curves
    print(f"Plotting best, worst, and median curves...")
    plt.figure(figsize=(12, 6))

    for i, idx in enumerate(indices_to_plot):
        x_vals = X_test[idx][4:]  # Get x-values (after first 4 params)
        true_y = Y_test[idx]
        predicted_y = Y_pred[idx]

        # Calculate RMSE for the current curve
        rmse = test_rmse_values[idx]

        # Parameters for the plot
        params = X_test[idx][:4]

        plt.subplot(1, 3, i + 1)
        plt.plot(x_vals, true_y, label="True Curve", marker="o", linestyle="-", color="blue")
        plt.plot(x_vals, predicted_y, label="Predicted Curve", marker="x", linestyle="--", color="red")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Curve {i+1}\nParams: {params}\nRMSE: {rmse:.4f}")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()

# Call the function to plot the best, worst, and median RMSE curves
Y_pred_final, _ = gp_model.predict(X_test, return_std=True)
plot_best_worst_median_curves(Y_pred_final, Y_test)

# ====== PLOT HISTOGRAM OF RMSE ======
def plot_rmse_histogram(Y_pred_final, Y_test):
    """Plot a histogram of RMSE values for the test set."""
    test_rmse_values = []
    print("Calculating RMSE values for the histogram...")
    for i in range(len(X_test)):
        true_y = Y_test[i]
        predicted_y = Y_pred_final[i]
        rmse = np.sqrt(np.mean((true_y - predicted_y) ** 2))
        test_rmse_values.append(rmse)
    
    print(f"RMSE values calculated for {len(X_test)} test samples.")

    # Plot histogram of RMSE values
    plt.figure(figsize=(8, 6))
    plt.hist(test_rmse_values, bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.title("Histogram of RMSE Values for Test Set")
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# Call the function to plot the histogram of RMSE values
plot_rmse_histogram(Y_pred_final, Y_test)
