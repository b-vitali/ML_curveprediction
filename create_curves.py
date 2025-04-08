import os
import random
import matplotlib.pyplot as plt
import numpy as np

# Folder to store files
output_folder = "curves"
os.makedirs(output_folder, exist_ok=True)

# Folder to save figures
figures_folder = "figures"
os.makedirs(figures_folder, exist_ok=True)

# Number of files to create
num_files = 200  # Change this to create more files

def generate_parameters():
    """Generate 6 random parameters with specific ranges for each."""
    return [
        random.uniform(45, 55),     # xmax
        random.uniform(15, 25),     # y0
        random.uniform(5, 15),      # x1
        random.uniform(35, 45),     # x2
        random.uniform(3.5, 4.5),   # m1
        random.uniform(-0.2, 0.2),  # m2
        random.uniform(-3.5, -2.5), # m3
    ]

def piecewise_function(xmax, y0, x1, x2, slope1, slope2, slope3):
    x = np.linspace(start=0, stop=xmax, num=1_000).reshape(-1, 1)
    y = np.zeros_like(x)
    
    # First segment: positive slope, intersects the origin
    y[x < x1] = y0 + slope1 * x[x < x1]
    
    # Second segment: slightly negative slope, continuous at x1
    intercept2 = y0 + slope1 * x1
    y[(x >= x1) & (x < x2)] = slope2 * (x[(x >= x1) & (x < x2)] - x1) + intercept2
    
    # Third segment: steep negative slope, continuous at x2
    intercept3 = slope2 * (x2 - x1) + intercept2
    y[x >= x2] = slope3 * (x[x >= x2] - x2) + intercept3
    
    return np.column_stack((x, y))


# Create files
for i in range(1, num_files + 1):
    params = generate_parameters()
    curve = piecewise_function(*params)
    
    filename = os.path.join(output_folder, f"curve_{i}.txt")
    
    with open(filename, "w") as f:
        f.write(f"Parameters: {params}\n")
        f.write("Points:\n")
        for x, y in curve:
            f.write(f"{x}, {y}\n")

print(f"Created {num_files} files in '{output_folder}'")

# Function to read and plot curves
def read_and_plot_curves(folder, save_path=None):
    """Reads curve files from a folder and plots them. Optionally saves the plot to a file."""
    plt.figure(figsize=(8, 6))
    
    for file in os.listdir(folder):
        if file.startswith("curve_") and file.endswith(".txt"):
            filepath = os.path.join(folder, file)
            
            with open(filepath, "r") as f:
                lines = f.readlines()
            
            # Extract parameters (just for reference)
            params = eval(lines[0].split(":")[1].strip())
            
            # Extract points
            points = [tuple(map(float, line.strip().split(", "))) for line in lines[2:]]
            x_vals, y_vals = zip(*points)
            
            plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=file)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Plotted Curves")
    plt.grid()
    
    # If save_path is provided, save the figure
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved as '{save_path}'")
    else:
        plt.show()

# Define the save path for the plot
save_plot_path = os.path.join(figures_folder, "curves.png")

# Read and plot the curves, saving the plot to the file
read_and_plot_curves(output_folder, save_path=save_plot_path)
