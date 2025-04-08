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
        "constant_value": 10,
        "constant_value_bounds": (1e-2, 1e4),
        "length_scale": 0.5,
        "length_scale_bounds": (0.1, 20.0)
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
            print("read parameters: ", params)

            points = [tuple(map(float, line.strip().split(", "))) for line in lines[2:]]
            x_vals, y_vals = zip(*points)
            x_vals, y_vals = np.array(x_vals), np.array(y_vals)

            #! Do i want to sample the whole curve OR real random?
            N = rng.poisson(num_points)
            """
            segment_size = len(x_vals) // num_points
            training_indices = np.array([
                rng.choice(np.arange(i * segment_size, (i + 1) * segment_size))
                for i in range(rng.poisson(num_points))
            ])
            print(rng.poisson(num_points))
            if len(x_vals) % num_points != 0:
                training_indices[-1] = len(x_vals) - 1
            
            """
            # Randomly select num_points indices from x_vals
            training_indices = rng.choice(len(x_vals), size=N, replace=False)
            training_indices = np.sort(training_indices)

            for point in range(N):
                X_sample = params + [x_vals[training_indices[point]]]
                #print("X_sample: ", X_sample)
                Y_sample = list(y_vals[training_indices])[point]
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
def plot_predicted_vs_true_curves_from_data(N=4, num_points_to_sample=100):
    files = sorted([f for f in os.listdir(config["data_folder"]) if f.startswith("curve_") and f.endswith(".txt")])[:N]
    print(f"Plotting predictions for the first {len(files)} curves...\n")

    plt.figure(figsize=(15, 5 * N))
    
    for i, file in enumerate(files):
        filepath = os.path.join(config["data_folder"], file)
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Read parameters
        params = eval(lines[0].split(":")[1].strip())
        print(params)
        # Load true x and y values
        points = [tuple(map(float, line.strip().split(", "))) for line in lines[2:]]
        x_vals_true, y_vals_true = zip(*points)
        x_vals_true, y_vals_true = np.array(x_vals_true), np.array(y_vals_true)

        # Generate evenly spaced x-values for prediction
        x_vals_interp = np.linspace(min(x_vals_true), max(x_vals_true), num_points_to_sample)
        expected_features = gp_model.n_features_in_
        X_input = [params[:expected_features - 1] + [x] for x in x_vals_interp]

        # Predict y using trained GP model
        Y_pred, sigma_pred = gp_model.predict(X_input, return_std=True)

        # Plot
        plt.subplot(N, 1, i + 1)
        plt.plot(x_vals_true, y_vals_true, label="True Curve", color="blue", linewidth=2)  # Smooth line
        plt.plot(x_vals_interp, Y_pred, label="Predicted Curve", color="red", linestyle="--")
        plt.scatter(x_vals_interp, Y_pred, color="red", marker="x", label="Predicted Points")  # Add predicted points
        plt.fill_between(x_vals_interp, Y_pred - 1.96 * sigma_pred, Y_pred + 1.96 * sigma_pred, alpha=0.3, color="red")
        plt.title(f"Curve {i+1} - {file}")
        plt.xlabel("x"), plt.ylabel("y")
        plt.grid(), plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config["figures_folder"], f"Predicted_vs_True_Curves.png"))
    plt.show()


# ========================== #
#        RUN PLOTTING        #
# ========================== #

plot_predicted_vs_true_curves_from_data(N=4)

