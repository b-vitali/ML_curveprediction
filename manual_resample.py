import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline, BarycentricInterpolator
import argparse
import json


# ========================== #
#      CONFIG DICTIONARY     #
# ========================== #
config = {
    "data_file": "data.txt",  # Default data file
    "interpolation": "linear",  # Default interpolation type
    "x_min": 0,  # Default starting x value
    "x_max": 60,  # Default ending x value
    "num_points": 40,  # Default number of points to sample
    "plot": {
        "show": True,  # Whether to show the plot
        "save": False,  # Whether to save the plot
        "file_name": "interpolated_plot.png"  # Default plot filename
    },
    "output_file": "output_data.txt"  # Default output file name
}

# ========================== #
#       SETUP & PARSING      #
# ========================== #

# Define argument parser
parser = argparse.ArgumentParser(description="Interpolation and resampling of data.")

# Add arguments for config
parser.add_argument('--data_file', type=str, help='Path to the data .txt file')
parser.add_argument('--interpolation', choices=['linear', 'cubic', 'quadratic', 'nearest', 'polynomial'],
                    default="linear", help="Type of interpolation to use (default: 'linear')")
parser.add_argument('--x_min', type=float, help='Starting x value')
parser.add_argument('--x_max', type=float, help='Ending x value')
parser.add_argument('--num_points', type=int, help='Number of points to sample')
parser.add_argument('--plot_show', type=bool, default=True, help='Whether to show the plot')
parser.add_argument('--plot_save', type=bool, default=False, help='Whether to save the plot')
parser.add_argument('--plot_file_name', type=str, default="interpolated_plot.png", help="Plot filename")
parser.add_argument('--output_file', type=str, default="output_data.txt", help="File to save output data")

# Parse the arguments
args = parser.parse_args()

# Override config with command line arguments (if provided)
if args.data_file is not None:
    config['data_file'] = args.data_file
if args.interpolation is not None:
    config['interpolation'] = args.interpolation
if args.x_min is not None:
    config['x_min'] = args.x_min
if args.x_max is not None:
    config['x_max'] = args.x_max
if args.num_points is not None:
    config['num_points'] = args.num_points
if args.plot_show is not None:
    config['plot']['show'] = args.plot_show
if args.plot_save is not None:
    config['plot']['save'] = args.plot_save
if args.plot_file_name is not None:
    config['plot']['file_name'] = args.plot_file_name
if args.output_file is not None:
    config['output_file'] = args.output_file

# Print out the config after parsing (with indentation)
print("Using config:")
print(json.dumps(config, indent=4))


# Function to parse the .txt file and extract data
def parse_txt_file(file_path):
    parameters = []
    points = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Read the "Parameters" section
        params = eval(lines[0].split(":")[1].strip())
        
        # Read the "Points" section
        points_start = lines.index("Points:\n") + 1
        for line in lines[points_start:]:
            x, y = map(float, line.strip().split(','))
            points.append((x, y))
    
    return params, np.array(points)


# Function to create interpolation function based on user input
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


# Main function to handle parsing, interpolation, and plotting
def main():
    # Parse the input file to extract parameters and data points
    parameters, points = parse_txt_file(config['data_file'])
    x_data, y_data = points[:, 0], points[:, 1]

    # Generate new x-values with the desired number of points
    new_x_values = np.linspace(config['x_min'], config['x_max'], config['num_points'])

    # Get the interpolation function based on user input
    interpolate_func = get_interpolation_function(x_data, y_data, config['interpolation'])

    # Evaluate y at new x values using the interpolation function
    new_y_values = interpolate_func(new_x_values)

    # Plot the original data, the resampled data, and the interpolated curve
    plt.plot(x_data, y_data, 'rx', label='Original data')  # Original points in red
    plt.plot(new_x_values, new_y_values, 'b-', label=f'{config["interpolation"].capitalize()} curve')  # Interpolated curve in blue
    plt.plot(new_x_values, new_y_values, 'go', label='Resampled data')  # Resampled points in green

    # Labels and title
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{config["interpolation"].capitalize()} Interpolation of x-y Data with Equal Spacing')

    # Show or save the plot based on config
    if config['plot']['show']:
        plt.show()
    if config['plot']['save']:
        plt.savefig(config['plot']['file_name'])
        print(f"Plot saved as {config['plot']['file_name']}.")

    # Save the output data (parameters, new x, and new y values) to the specified output file
    with open(config['output_file'], 'w') as output_file:
        output_file.write(f"Parameters: {parameters}\n")
        output_file.write("Points:\n")
        for x, y in zip(new_x_values, new_y_values):
            output_file.write(f"{x}, {y}\n")
        
    print(f"Data saved to {config['output_file']}.")

    
# Run the main function if the script is executed
if __name__ == "__main__":
    main()
