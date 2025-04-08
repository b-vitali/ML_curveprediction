import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline, BarycentricInterpolator
import argparse
import json
import glob

# ========================== #
#      CONFIG DICTIONARY     #
# ========================== #
config = {
    "input_dir": "curves",      # Folder containing input .txt files
    "output_dir": "resampled_curves", # Folder to save resampled .txt files
    "interpolation": "linear",
    "x_min": 0,
    "x_max": 55,
    "num_points": 40,
    "plot": {
        "show": False,
        "save": False,
        "file_name": "plot.png"
    }
}

# ========================== #
#       ARGUMENT PARSING     #
# ========================== #
parser = argparse.ArgumentParser(description="Batch interpolation and resampling of curve data.")

parser.add_argument('--input_dir', type=str, help='Input folder with curve_*.txt files')
parser.add_argument('--output_dir', type=str, help='Output folder to save resampled files')
parser.add_argument('--interpolation', choices=['linear', 'cubic', 'quadratic', 'nearest', 'polynomial'], help="Interpolation type")
parser.add_argument('--x_min', type=float, help='Min x value')
parser.add_argument('--x_max', type=float, help='Max x value')
parser.add_argument('--num_points', type=int, help='Number of points to sample')

args = parser.parse_args()

# Override config
if args.input_dir: config['input_dir'] = args.input_dir
if args.output_dir: config['output_dir'] = args.output_dir
if args.interpolation: config['interpolation'] = args.interpolation
if args.x_min is not None: config['x_min'] = args.x_min
if args.x_max is not None: config['x_max'] = args.x_max
if args.num_points: config['num_points'] = args.num_points

print("Using config:")
print(json.dumps(config, indent=4))

# ========================== #
#       HELPER FUNCTIONS     #
# ========================== #

def parse_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        parameters = eval(lines[0].split(":", 1)[1].strip())
        points_start = lines.index("Points:\n") + 1
        points = [tuple(map(float, line.strip().split(','))) for line in lines[points_start:]]
    return parameters, np.array(points)

def get_interpolation_function(x_data, y_data, interpolation_type):
    if interpolation_type == 'linear':
        return interp1d(x_data, y_data, kind='linear', fill_value="extrapolate")
    elif interpolation_type == 'cubic':
        return CubicSpline(x_data, y_data, bc_type='natural')
    elif interpolation_type == 'quadratic':
        return interp1d(x_data, y_data, kind='quadratic', fill_value="extrapolate")
    elif interpolation_type == 'nearest':
        return interp1d(x_data, y_data, kind='nearest', fill_value="extrapolate")
    elif interpolation_type == 'polynomial':
        return BarycentricInterpolator(x_data, y_data)
    else:
        raise ValueError(f"Unsupported interpolation type: {interpolation_type}")

def resample_and_save(file_path, output_path):
    parameters, points = parse_txt_file(file_path)
    x_data, y_data = points[:, 0], points[:, 1]
    
    new_x = np.linspace(config['x_min'], config['x_max'], config['num_points'])
    interpolator = get_interpolation_function(x_data, y_data, config['interpolation'])
    new_y = interpolator(new_x)

    with open(output_path, 'w') as f:
        f.write(f"Parameters: {parameters}\n")
        f.write("Points:\n")
        for x, y in zip(new_x, new_y):
            f.write(f"{x}, {y}\n")
    print(f"Saved: {output_path}")

# ========================== #
#           MAIN             #
# ========================== #

def main():
    os.makedirs(config['output_dir'], exist_ok=True)
    input_pattern = os.path.join(config['input_dir'], "curve_*.txt")
    input_files = sorted(glob.glob(input_pattern))

    if not input_files:
        print("No input files found.")
        return

    for file_path in input_files:
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        resampled_name = name_without_ext + "_rs.txt"
        output_path = os.path.join(config['output_dir'], resampled_name)
        resample_and_save(file_path, output_path)

if __name__ == "__main__":
    main()
