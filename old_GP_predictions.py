import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(34)
random_seed = 34

# Folder containing curve files
data_folder = "curves"

# Folder to save figures
figures_folder = "figures"
os.makedirs(figures_folder, exist_ok=True)

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = random_seed)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Define Gaussian Process model with RBF kernel
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))  # Constant kernel * RBF kernel
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Training the Gaussian Process model
print("Training the Gaussian Process model...")

# GP needs to be fitted on each individual curve
gp_model.fit(X_train, Y_train)
print(f"Training complete! Optimized kernel: {gp_model.kernel_}")

# Predictions and evaluation
print("Making predictions on the test set...")

# Make predictions on the test set
Y_pred, sigma = gp_model.predict(X_test, return_std=True)  # Standard deviation (uncertainty) can also be returned

# Compute RMSE
test_mse = np.mean((Y_pred - Y_test) ** 2)
test_rmse = np.sqrt(test_mse)

print(f"Test RMSE: {test_rmse:.6f}")

# ====== PLOT BEST, WORST, AND MEDIAN RMSE CURVES ======
def plot_best_worst_median_curves():
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

    # Save the plot
    plt.savefig(os.path.join(figures_folder, "GP_best_worst_median_rmse_curves.png"))
    plt.show()

# Call the function to plot the best, worst, and median RMSE curves
plot_best_worst_median_curves()

# ====== PLOT DIFFERENCE AND DIFFERENCE % ======
def plot_differences():
    """Plot the differences between expected and true y, and differences divided by uncertainty for all test samples."""

    # Calculate differences between predicted and true values
    differences = Y_pred - Y_test
    print("Calculating differences between predicted and true values...")

    # Divide differences by uncertainty (standard deviation)
    differences_percentage = differences / Y_test
    print("Calculating differences divided by uncertainty...")

    # Plotting the differences and differences divided by uncertainty
    plt.figure(figsize=(14, 6))

    # Plot raw differences (predicted - true)
    plt.subplot(1, 2, 1)
    for i in range(len(X_test)):
        plt.plot(X_test[i][4:], differences[i], label=f"Sample {i+1}", marker="o", linestyle="--", alpha=0.6)
    plt.xlabel("X")
    plt.ylabel("Difference (Predicted - True)")
    plt.title("Differences Between Predicted and True Y")
    plt.grid()

    # Plot differences divided by uncertainty
    plt.subplot(1, 2, 2)
    for i in range(len(X_test)):
        plt.plot(X_test[i][4:], differences_percentage[i], label=f"Sample {i+1}", marker="o", linestyle="--", alpha=0.6)
    plt.xlabel("X")
    plt.ylabel("Difference in %")
    plt.title("Percentual difference")
    plt.grid()

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(figures_folder, "GP_differences.png"))
    plt.show()

# Call the function to plot the differences and differences divided by uncertainty for all test samples
plot_differences()


# ====== PLOT HISTOGRAM OF RMSE ======
def plot_rmse_histogram():
    """Plot a histogram of RMSE values for the test set."""
    test_rmse_values = []
    print("Calculating RMSE values for the histogram...")
    for i in range(len(X_test)):
        true_y = Y_test[i]
        predicted_y = Y_pred[i]
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

    # Save the histogram
    plt.savefig(os.path.join(figures_folder, "GP_rmse_histogram.png"))
    plt.show()

# Call the function to plot and save the histogram of RMSE values
plot_rmse_histogram()
