import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import scipy.interpolate as interp
import math

# Function to read griddata from a CSV or TXT file and perform interpolation
def read_and_interpolate_griddata(file_path, x_grid, y_grid):
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Extract x, y, and z values
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values

    # Create a meshgrid from the provided x and y grids
    X, Y = np.meshgrid(x_grid, y_grid)

    # Interpolate the data using griddata function
    griddata = interp.griddata((x, y), z, (X, Y), method='linear', fill_value=np.nan)

    return griddata

# Function for calculating average over wheel bounds, handling NaN values
def calculate_average_over_rectangle(griddata, x_grid, y_grid, bounds):
    xmin, xmax, ymin, ymax = bounds
    x_mask = (x_grid >= xmin) & (x_grid <= xmax)
    y_mask = (y_grid >= ymin) & (y_grid <= ymax)

    selected_x_indices = np.where(x_mask)[0]
    selected_y_indices = np.where(y_mask)[0]

    if selected_x_indices.size == 0 or selected_y_indices.size == 0:
        return 0  # Handle case where wheel is outside grid by returning 0

    selected_values = griddata[np.ix_(selected_y_indices, selected_x_indices)]
    average_value = np.nanmean(selected_values)  # Avoid NaN propagation
    
    return average_value

# Function for plotting griddata with wheel bounds and average values
def plot_griddata_with_wheels(griddata, x_grid, y_grid, wheel_bounds, average_values, second_lane=False, second_lane_bounds=None, second_lane_avg_values=None):
    X, Y = np.meshgrid(x_grid, y_grid)
    
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, griddata, cmap='viridis')
    plt.colorbar(contour, label='Griddata Value')

    # Plot the wheel bounds and annotate with average values
    for bounds, avg_value in zip(wheel_bounds, average_values):
        xmin, xmax, ymin, ymax = bounds[:4]  # Unpack only the first four elements
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='red', lw=2)

        # Annotate the plot with the average value at the center of each wheel
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        plt.text(x_center, y_center, f'{avg_value:.2f}', color='white', fontsize=10, ha='center', va='center')

    if second_lane:
        # Plot bounds for the second lane with another vehicle
        for bounds, avg_value in zip(second_lane_bounds, second_lane_avg_values):
            xmin, xmax, ymin, ymax = bounds[:4]  # Unpack only the first four elements
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='blue', lw=2)

            # Annotate the plot with the average value at the center of each wheel in the second lane
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            plt.text(x_center, y_center, f'{avg_value:.2f}', color='white', fontsize=10, ha='center', va='center')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Griddata with Wheel Bounds and Average Values')
    plt.gca().set_aspect('equal', adjustable='box')
    st.pyplot(plt)



# Updated function for calculating wheel bounds based on centering, rotation, and weight factors
def get_vehicle_wheel_bounds(vehicle_type, span_length, wheel_distance, center_method, orientation, wheel_to_center=None):
    wheel_width = 0.3/span_length
    wheel_length = 0.2/span_length
    wheel_distance = wheel_distance/span_length
    
    if vehicle_type == 'Typfordon b':
        wheels = [
            (-0.5/span_length, -0.5*wheel_distance, 0.22),  # Front left
            (0.5/span_length, -0.5*wheel_distance, 0.22),   # Front right
            (-0.5/span_length, 0.5*wheel_distance, 0.22),   # Rear left
            (0.5/span_length, 0.5*wheel_distance, 0.22)     # Rear right
        ]

    elif vehicle_type == 'Typfordon c':
        wheels = [
            (-0.65/span_length, -0.5*wheel_distance, 0.25),  # Front left
            (0.65/span_length, -0.5*wheel_distance, 0.25),   # Front right
            (-0.65/span_length, 0.5*wheel_distance, 0.25),   # Rear left
            (0.65/span_length, 0.5*wheel_distance, 0.25)     # Rear right
        ]

    elif vehicle_type == 'Typfordon d':
        wheels = [
            (-0.9/span_length, -0.5*wheel_distance, 0.275),  # Front left
            (0.9/span_length, -0.5*wheel_distance, 0.275),   # Front right
            (-0.9/span_length, 0.5*wheel_distance, 0.275),   # Rear left
            (0.9/span_length, 0.5*wheel_distance, 0.275)     # Rear right
        ]
        
    elif vehicle_type == 'Typfordon e':
        wheels = [
            (-1/span_length, -0.5*wheel_distance, 0.195),  # Front left
            (1/span_length, -0.5*wheel_distance, 0.195),   # Front right
            (0, 0.5*wheel_distance, 0.195),   # Rear left
            (0, -0.5*wheel_distance, 0.195),     # Rear right
            (-1/span_length, 0.5*wheel_distance, 0.195),   # Rear left
            (1/span_length, 0.5*wheel_distance, 0.195)     # Rear right
        ]
        
    elif vehicle_type == 'Typfordon f':
        wheels = [
            (-1.3/span_length, -0.5*wheel_distance, 0.22),  # Front left
            (1.3/span_length, -0.5*wheel_distance, 0.22),   # Front right
            (0, 0.5*wheel_distance, 0.22),   # Rear left
            (0, -0.5*wheel_distance, 0.22),     # Rear right
            (-1.3/span_length, 0.5*wheel_distance, 0.22),   # Rear left
            (1.3/span_length, 0.5*wheel_distance, 0.22)     # Rear right
        ]
    
    elif vehicle_type == 'Service vehicle':
        wheels = [
            (-1.5/span_length, -0.65/span_length, 0.5),  # Front left
            (1.5/span_length, -0.65/span_length, 0.25),   # Front right
            (-1.5/span_length, 0.65/span_length, 0.5),   # Rear left
            (1.5/span_length, 0.65/span_length, 0.25)     # Rear right
        ]
        wheel_width = 0.4/span_length
        wheel_length = 0.4/span_length

    elif vehicle_type == 'LM1':
        wheels = [
            (-0.6/span_length, -0.5*wheel_distance, 0.5),  # Front left
            (0.6/span_length, -0.5*wheel_distance, 0.5),   # Front right
            (-0.6/span_length, 0.5*wheel_distance, 0.5),   # Rear left
            (0.6/span_length, 0.5*wheel_distance, 0.5)     # Rear right
        ]
        wheel_width = 0.4/span_length
        wheel_length = 0.4/span_length

    elif vehicle_type == 'LM2':
        wheels = [
            (0, -0.5*wheel_distance, 0.5),  # Front left
            (0, 0.5*wheel_distance, 0.5),   # Front right
        ]    
        wheel_width = 0.6/span_length
        wheel_length = 0.35/span_length

    # Rotate the vehicle 90 degrees if selected
    if orientation == '90° Rotation':
        wheels = [(y, -x, w) for x, y, w in wheels]
        wheel_width, wheel_length = wheel_length, wheel_width  # Swap width and length for rotation

    # Adjust positions based on centering method
    if center_method == 'Center Around Origin':
        centered_wheels = wheels
    elif center_method == 'Center Around Wheel' and wheel_to_center is not None:
        x_offset, y_offset, _ = wheels[wheel_to_center]
        centered_wheels = [(x - x_offset, y - y_offset, w) for x, y, w in wheels]
    
    wheel_bounds = [
        (x - wheel_length/2, x + wheel_length/2, y - wheel_width/2, y + wheel_width/2, w)
        for x, y, w in centered_wheels
    ]

    return wheel_bounds

# Streamlit App Setup
st.title("Griddata Analysis App with Vehicle Wheels and Multiple Lanes")
st.sidebar.header("Settings")

# Define span length
span_length = st.sidebar.number_input("Span length", value=5.00, step=0.01)

# Define the directory where the griddata files are stored
griddata_dir = "griddata_files"  # Folder where CSV files are stored

# Get a list of all CSV files in the directory
griddata_files = [f for f in os.listdir(griddata_dir) if f.endswith('.csv')]

# Sidebar dropdown for selecting a griddata file
selected_file = st.sidebar.selectbox("Choose a griddata file", griddata_files)

# Load the selected griddata file
file_path = os.path.join(griddata_dir, selected_file)

x_grid = np.linspace(-1.5, 1.5, 600)  # Example grid
y_grid = np.linspace(-0.5, 0.5, 200)  # Example grid

# Load and interpolate griddata
griddata = read_and_interpolate_griddata(file_path, x_grid, y_grid)

# Sidebar dropdown for selecting a griddata file
B_load = st.sidebar.number_input("Write a axle/boggie load in kN", value=200)

# Dropdown for vehicle type
vehicle_type = st.sidebar.selectbox(
    "Select Vehicle Type",
    ("Typfordon b", "Typfordon c", "Typfordon d", "Typfordon e", "Typfordon f","LM1","LM2","Service vehicle")
)
# Dropdown for wheel distance
wheel_distance = st.sidebar.selectbox(
    "Select distance between wheels",
    (1.7, 2.0, 2.3)
)

# Dropdown for centering method
center_method = st.sidebar.selectbox(
    "Select Centering Method",
    ("Center Around Origin", "Center Around Wheel")
)

# Dropdown for wheel selection (only if centering around a specific wheel)
wheel_to_center = None
if center_method == 'Center Around Wheel':
    wheel_to_center = st.sidebar.selectbox(
        "Select Wheel to Center",
        list(range(4 if vehicle_type in ['Typfordon b', 'Typfordon c', 'Typfordon d'] else 6 if vehicle_type in ['Typfordon e', 'Typfordon f'] else 8))
    )

# Dropdown for vehicle orientation
orientation = st.sidebar.selectbox(
    "Select Vehicle Orientation",
    ("0° Rotation", "90° Rotation")
)

# Option to enable/disable second lane
second_lane_enabled = st.sidebar.checkbox("Enable Second Lane")

# Get vehicle wheel bounds based on the selected vehicle type, centering method, orientation, and scale factor
wheel_bounds = get_vehicle_wheel_bounds(vehicle_type, span_length, wheel_distance, center_method, orientation, wheel_to_center)

# Calculate average values for each wheel
average_values = [
    calculate_average_over_rectangle(griddata, x_grid, y_grid, bounds[:4]) * bounds[4]*B_load/8/math.pi
    for bounds in wheel_bounds
]

# Sum the average values for the first lane vehicle
total_average = np.nansum(average_values)

lane_distance=3/span_length
# Handle second lane vehicle, applying 3.0 m offset and 0.8 weight factor
if second_lane_enabled:
    if vehicle_type == 'LM1':
        load_factor = 2/3
    else:
        load_factor = 0.8
    if orientation == '90° Rotation':
        # Offset the vehicle in the X-axis if rotated
        second_lane_bounds = [(xmin + lane_distance, xmax + lane_distance, ymin, ymax, load_factor * 0.8) for xmin, xmax, ymin, ymax, weight in wheel_bounds]
    else:
        # Offset the vehicle in the Y-axis if not rotated
        second_lane_bounds = [(xmin, xmax, ymin + lane_distance, ymax + lane_distance, load_factor * 0.8) for xmin, xmax, ymin, ymax, weight in wheel_bounds]

    # Calculate average values for the second lane vehicle
    second_lane_average_values = [
        calculate_average_over_rectangle(griddata, x_grid, y_grid, bounds[:4]) * bounds[4]*B_load/8/math.pi
        for bounds in second_lane_bounds
    ]
    
    # Sum the average values for the second lane vehicle
    total_average += np.nansum(second_lane_average_values)
else:
    second_lane_bounds = []
    second_lane_average_values = []

# Plot the griddata with the wheel bounds and display the average values per wheel
plot_griddata_with_wheels(griddata, x_grid, y_grid, wheel_bounds, average_values, second_lane_enabled, second_lane_bounds, second_lane_average_values)

# Display the total average value as text
st.write(f"Total Average Value: {total_average:.2f}")

