import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(34)
np.random.seed(34)

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
            
            # Convert to tensors
            X.append(params + list(x_vals))  # 4 params + 50 x-values (input size = 54)
            Y.append(list(y_vals))           # 50 y-values (output size = 50)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Load data
print("Loading dataset...")
X, Y = load_dataset(data_folder)
print(f"Dataset loaded! Total samples: {len(X)}")

# Split into training (80%) and testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Define Neural Network model
class CurvePredictor(nn.Module):
    def __init__(self):
        super(CurvePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(54, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 50)  # Output 50 y-values
        )

    def forward(self, x):
        return self.model(x)

# Instantiate model, loss function, and optimizer
model = CurvePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1500
batch_size = 16
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

loss_history = []
rmse_history = []

print("Training the model...")
for epoch in range(epochs):
    epoch_loss = 0

    # Mini-batch training
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Compute average loss for epoch
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    
    # Compute RMSE on test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_mse = criterion(test_predictions, Y_test).item()
        test_rmse = np.sqrt(test_mse)
        rmse_history.append(test_rmse)
    
    model.train()  # Switch back to training mode

    # Print updates
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d}/{epochs}, Loss: {avg_loss:.6f}, Test RMSE: {test_rmse:.6f}")

print("Training complete!")

# Final evaluation
model.eval()
with torch.no_grad():
    final_predictions = model(X_test)
    final_mse = criterion(final_predictions, Y_test).item()
    final_rmse = np.sqrt(final_mse)

print(f"Final Test RMSE: {final_rmse:.6f}")

# Plot Loss and RMSE Over Epochs on the same plot with two y-axes
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot the training loss on the primary y-axis
ax1.plot(range(epochs), loss_history, label="Training Loss (MSE)", color="blue")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss (MSE)", color="blue")
ax1.set_yscale('log')  # Set log scale for primary y-axis
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True)

# Create a secondary y-axis to plot RMSE
ax2 = ax1.twinx()
ax2.plot(range(epochs), rmse_history, label="Test RMSE", color="red")
ax2.set_ylabel("RMSE", color="red")
ax2.set_yscale('log')  # Set log scale for primary y-axis
ax2.tick_params(axis="y", labelcolor="red")

# Title and legends
plt.title("Training Loss (MSE) and Test RMSE Over Epochs")
fig.tight_layout()

# Save the plot
plt.savefig(os.path.join(figures_folder, "NN_training_loss_rmse.png"))
plt.show()

def plot_best_worst_median_curves():
    """Plot the best, worst, and median RMSE curves, with true vs predicted values and RMSE."""

    # Calculate RMSE for each test sample
    test_rmse_values = []
    for i in range(len(X_test)):
        true_y = Y_test[i].numpy()
        predicted_y = final_predictions[i].numpy()
        rmse = np.sqrt(np.mean((true_y - predicted_y) ** 2))
        test_rmse_values.append(rmse)

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
    plt.figure(figsize=(12, 6))

    for i, idx in enumerate(indices_to_plot):
        x_vals = X_test[idx][4:].numpy()  # Get x-values (after first 4 params)
        true_y = Y_test[idx].numpy()
        predicted_y = final_predictions[idx].numpy()

        # Calculate RMSE for the current curve
        rmse = test_rmse_values[idx]

        # Parameters for the plot
        params = X_test[idx][:4].numpy()

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
    plt.savefig(os.path.join(figures_folder, "NN_best_worst_median_rmse_curves.png"))
    plt.show()

# Call the function to plot the best, worst, and median RMSE curves
plot_best_worst_median_curves()

# ====== PLOT HISTOGRAM OF RMSE ======
def plot_rmse_histogram():
    """Plot a histogram of RMSE values for the test set."""
    test_rmse_values = []
    for i in range(len(X_test)):
        true_y = Y_test[i].numpy()
        predicted_y = final_predictions[i].numpy()
        rmse = np.sqrt(np.mean((true_y - predicted_y) ** 2))
        test_rmse_values.append(rmse)

    # Plot histogram of RMSE values
    plt.figure(figsize=(8, 6))
    plt.hist(test_rmse_values, bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.title("Histogram of RMSE Values for Test Set")
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save the histogram
    plt.savefig(os.path.join(figures_folder, "NN_rmse_histogram.png"))
    plt.show()

# Call the function to plot and save the histogram of RMSE values
plot_rmse_histogram()

