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
num_files = 100  # Change this to create more files
num_points = 50  # Change this to create more points

def generate_parameters():
    """Generate an array of 4 random parameters. Two of them will be used in the linear equation."""
    return [random.uniform(0, 10) for _ in range(4)]

def compute_curve(params):
    a, b, c, d = params
    """Generate (x, y) points using a polynomial with some random noise."""
    points = [(x, a + b*x + c*x*x + random.uniform(-0.2, 0.2)) for x in range(num_points)]
    return points

# Create files
for i in range(1, num_files + 1):
    params = generate_parameters()
    curve = compute_curve(params)
    
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
