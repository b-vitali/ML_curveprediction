import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import argparse

# ========================== #
#      CONFIG DICTIONARY     #
# ========================== #
config = {
    "random_seed": 34,
    "num_points": 20,
    "max_curves": None,
    "data_folder": "curves",
    "figures_folder": "figures",
    "kernel": {
        "constant_value": 100,
        "constant_value_bounds": (1e1, 1e5),
        "length_scale": 5.0,
        "length_scale_bounds": (1.0, 50.0)
    },
    "gp": {
        "n_restarts_optimizer": 10
    },
    "test_size": 0.2
}


# ========================== #
#       SETUP & PARSING      #
# ========================== #

# Define argument parser
parser = argparse.ArgumentParser(description="Gaussian Process Curve Fitting")

# Add arguments for config
parser.add_argument('--random_seed', type=int, help='Random seed for reproducibility')
parser.add_argument('--num_points', type=int, help='Override number of samples to pick per curve')
parser.add_argument('--data_folder', type=str, help='Folder containing the data files')
parser.add_argument('--figures_folder', type=str, help='Folder to save figures')
parser.add_argument('--constant_value', type=float, help='Constant kernel value')
parser.add_argument('--constant_value_bounds', type=str, help='Bounds for constant kernel value, format (min, max)')
parser.add_argument('--length_scale', type=float, help='Length scale for RBF kernel')
parser.add_argument('--length_scale_bounds', type=str, help='Bounds for length scale, format (min, max)')
parser.add_argument('--n_restarts_optimizer', type=int, help='Number of restarts for the optimizer')
parser.add_argument('--test_size', type=float, help='Proportion of the dataset to include in the test split')
parser.add_argument('--max_curves', type=int, help='Maximum number of curves to include from the dataset')

# Parse the arguments
args = parser.parse_args()

# Override config with command line arguments (if provided)
if args.random_seed is not None:
    config['random_seed'] = args.random_seed
if args.num_points is not None:
    config['num_points'] = args.num_points
if args.data_folder is not None:
    config['data_folder'] = args.data_folder
if args.figures_folder is not None:
    config['figures_folder'] = args.figures_folder
if args.constant_value is not None:
    config['kernel']['constant_value'] = args.constant_value
if args.constant_value_bounds is not None:
    config['kernel']['constant_value_bounds'] = tuple(map(float, args.constant_value_bounds.strip('()').split(',')))
if args.length_scale is not None:
    config['kernel']['length_scale'] = args.length_scale
if args.length_scale_bounds is not None:
    config['kernel']['length_scale_bounds'] = tuple(map(float, args.length_scale_bounds.strip('()').split(',')))
if args.n_restarts_optimizer is not None:
    config['gp']['n_restarts_optimizer'] = args.n_restarts_optimizer
if args.test_size is not None:
    config['test_size'] = args.test_size
if args.max_curves is not None:
    config['max_curves'] = args.max_curves

# Print out the config after parsing (with indentation)
import json
print("Using config:")
print(json.dumps(config, indent=4))

# ========================== #
#        SETUP & DATA        #
# ========================== #

# Set random seed
np.random.seed(config["random_seed"])
rng = np.random.RandomState(config["random_seed"])

# Create output folder
os.makedirs(config["figures_folder"], exist_ok=True)

# ========================== #
#        DATA LOADING        #
# ========================== #

num_param = 0

def list_shape(lst):
    if isinstance(lst, list):
        return [len(lst)] + list_shape(lst[0]) if lst else []
    else:
        return []


def load_dataset(folder, num_points, max_curves=None):
    X, Y = [], []
    curve_count = 0    
    print("Loading data from folder:", folder)

    for file in os.listdir(folder):
        if file.startswith("curve_") and file.endswith(".txt"):
            # Stop if max_curves is reached
            if max_curves is not None and curve_count >= max_curves:
                break  
            curve_count += 1
            filepath = os.path.join(folder, file)
            with open(filepath, "r") as f:
                lines = f.readlines()

            
            global num_param  
            params = eval(lines[0].split(":")[1].strip())
            num_param = len(params)
            
            points = [tuple(map(float, line.strip().split(", "))) for line in lines[2:]]
            x_vals, y_vals = zip(*points)
            x_vals, y_vals = np.array(x_vals), np.array(y_vals)

            #! Do i want to sample the whole curve OR real random?
            segment_size = len(x_vals) // num_points
            training_indices = np.array([
                rng.choice(np.arange(i * segment_size, (i + 1) * segment_size))
                for i in range(num_points)
            ])

            if len(x_vals) % num_points != 0:
                training_indices[-1] = len(x_vals) - 1
            
            """
            global rng
            # Randomly select num_points indices from x_vals
            training_indices = rng.choice(len(x_vals), size=num_points, replace=False)
            training_indices = np.sort(training_indices)
            """
            X_sample = params + list(x_vals[training_indices])
            Y_sample = list(y_vals[training_indices])
            X.append(X_sample)
            Y.append(Y_sample)

    print(f"Data loaded: {len(X)} samples")
    return np.array(X), np.array(Y)

print("\nRunning: load_dataset ...")
X, Y = load_dataset(config["data_folder"], config["num_points"], config.get("max_curves"))
print(f"Dataset loaded! Total samples: {len(X)}\n")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=config["test_size"], random_state=config["random_seed"]
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ========================== #
#      MODEL INITIALIZE      #
# ========================== #

kcfg = config["kernel"]
kernel = C(kcfg["constant_value"], kcfg["constant_value_bounds"]) * RBF(kcfg["length_scale"], kcfg["length_scale_bounds"])
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=config["gp"]["n_restarts_optimizer"])

print(f"Define kernel as: k = C*RBF")
print("Training the Gaussian Process model...\n")
gp_model.fit(X_train, Y_train)
print(f"Training complete! Optimized kernel: {gp_model.kernel_}")

# ========================== #
#     PREDICTION & METRICS   #
# ========================== #

print("Making predictions on the test set...\n")
Y_pred, sigma = gp_model.predict(X_test, return_std=True)
test_rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
print(f"Test RMSE: {test_rmse:.6f}\n")

# ========================== #
#       PLOTTING FUNCS       #
# ========================== #

def plot_best_worst_median_curves():
    test_rmse_values = [
        np.sqrt(np.mean((Y_test[i] - Y_pred[i]) ** 2)) for i in range(len(X_test))
    ]

    sorted_indices = np.argsort(test_rmse_values)
    best_idx, worst_idx = sorted_indices[0], sorted_indices[-1]
    median_val = np.median(test_rmse_values)
    median_idx = min(range(len(test_rmse_values)), key=lambda i: abs(test_rmse_values[i] - median_val))
    indices_to_plot = [best_idx, worst_idx, median_idx]

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices_to_plot):
        x_vals = X_test[idx][num_param:]
        true_y, pred_y = Y_test[idx], Y_pred[idx]
        sigma_vals = sigma[idx]
        ci_upper = pred_y + 1.96 * sigma_vals
        ci_lower = pred_y - 1.96 * sigma_vals
        rmse = test_rmse_values[idx]
        params = X_test[idx][:num_param]

        plt.subplot(1, 3, i + 1)
        plt.plot(x_vals, true_y, label="True", marker="o", color="blue")
        plt.plot(x_vals, pred_y, label="Pred", marker="x", linestyle="--", color="red")
        plt.fill_between(x_vals, ci_lower, ci_upper, color="gray", alpha=0.3)
        plt.title(f"Curve {i+1}\nRMSE: {rmse:.4f}")
        plt.legend(), plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(config["figures_folder"], "GP_best_worst_median_rmse_curves_with_ci.png"))
    plt.show()

def plot_differences():
    differences = Y_pred - Y_test
    diffs_percent = differences / sigma

    plt.figure(figsize=(14, 6))
    for i in range(len(X_test)):
        x_vals = X_test[i][num_param:]
        plt.subplot(1, 2, 1)
        plt.plot(x_vals, differences[i], alpha=0.6)
        plt.subplot(1, 2, 2)
        plt.plot(x_vals, diffs_percent[i], alpha=0.6)

    plt.subplot(1, 2, 1)
    plt.title("Predicted - True Y"), plt.grid(), plt.xlabel("X"), plt.ylabel("Diff")
    plt.subplot(1, 2, 2)
    plt.title("Diff / Sigma"), plt.grid(), plt.xlabel("X"), plt.ylabel("Z-score")

    plt.tight_layout()
    plt.savefig(os.path.join(config["figures_folder"], "GP_differences.png"))
    plt.show()

def plot_rmse_histogram():
    test_rmse_values = [
        np.sqrt(np.mean((Y_test[i] - Y_pred[i]) ** 2)) for i in range(len(X_test))
    ]

    plt.figure(figsize=(8, 6))
    plt.hist(test_rmse_values, bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.title("Histogram of RMSEs"), plt.xlabel("RMSE"), plt.ylabel("Count"), plt.grid()
    plt.savefig(os.path.join(config["figures_folder"], "GP_rmse_histogram.png"))
    plt.show()

def plot_curves_from_n1_to_n2(n1=0, n2=5):
    n2 = min(n2, len(X_test))
    cmap = plt.colormaps.get_cmap("tab10")
    colors = [cmap(i) for i in range(n2 - n1)]

    plt.figure(figsize=(14, 6))
    for i in range(n1, n2):
        x_vals = X_test[i][num_param:]
        true_y = Y_test[i]
        pred_y = Y_pred[i]
        ci_upper = pred_y + 1.96 * sigma[i]
        ci_lower = pred_y - 1.96 * sigma[i]
        color = colors[i - n1]

        plt.plot(x_vals, true_y, label=f"True {i+1}", color=color, marker="o")
        plt.plot(x_vals, pred_y, label=f"Pred {i+1}", color=color, linestyle="--", marker="x")
        plt.fill_between(x_vals, ci_lower, ci_upper, color=color, alpha=0.3)

    plt.xlabel("X"), plt.ylabel("Y")
    plt.title(f"Curves {n1+1} to {n2}")
    plt.legend(), plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(config["figures_folder"], f"Curves_{n1+1}_to_{n2}.png"))
    plt.show()

# ========================== #
#        RUN PLOTTING        #
# ========================== #

plot_best_worst_median_curves()
plot_differences()
plot_rmse_histogram()
plot_curves_from_n1_to_n2(0, 5)
