from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
import math


class HistogramData(NamedTuple):
    data: dict[str, list[float]]
    name: str


class PlotData(NamedTuple):
    data: dict[str, float]
    name: str


def plot_histogram(data, num_bars, filename):
    """
    Plots a histogram from an array of numbers, with N bars, and saves it to a file.

    Parameters:
    - data: Array of numbers to plot.
    - num_bars: The number of bars in the histogram.
    - filename: The file path where the image will be saved.
    """
    # Set the range for the X-axis: from 0 to the max value in the data
    x_min, x_max = 0, max(data)

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=num_bars, range=(x_min, x_max), edgecolor="black")

    # Label the axes
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram")

    # Save the plot to the specified file
    plt.savefig(filename)
    plt.close()


def plot_histograms(data_dict, filename, num_bins=20, max_cols=5, x_max=None):
    """
    Plots multiple histograms based on a nested dictionary structure.

    Parameters:
    - data_dict: A nested dictionary where:
        - First-level keys are plot titles.
        - First-level values are dictionaries with:
            - Second-level keys as dataset names (for the legend).
            - Second-level values as arrays of data to plot.
    - filename: The file path where the image will be saved.
    - num_bins: The number of bins in the histograms.
    """
    # Determine the number of plots
    num_plots = len(data_dict)
    num_cols = min(num_plots, max_cols)
    num_rows = math.ceil(num_plots / max_cols)

    # Create the subplots grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))

    # Flatten axes array for easy indexing if multiple plots
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Iterate over the plots
    for idx, histogram_data in enumerate(data_dict):
        plot_title = histogram_data.name
        datasets = histogram_data.data

        ax = axes[idx]
        data_arrays = []
        labels = []

        # Collect data and labels
        for dataset_name, data in datasets.items():
            data_arrays.append(data)
            labels.append(dataset_name)

        # Determine the X-axis range (from 0 to the maximum value in the data)
        x_min = 0

        if x_max is None:
            x_max = max([max(data) for data in data_arrays if len(data) > 0], default=1)

        # Plot the histograms
        ax.hist(
            data_arrays, bins=num_bins, range=(x_min, x_max), label=labels, alpha=0.7
        )

        # Set plot attributes
        ax.set_title(plot_title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend()

    # Hide any unused subplots
    total_subplots = num_rows * num_cols
    if total_subplots > num_plots:
        for idx in range(num_plots, total_subplots):
            fig.delaxes(axes[idx])

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_bar_charts(data_array: list[PlotData], filename: str):
    """
    Plots a bar chart for each category with sub-category colored bars and a legend.

    Parameters:
    - data_array: List of PlotData objects, each representing a category with sub-categories.
    - filename: The file path where the image will be saved.
    """
    # Extract all unique sub-categories across all PlotData
    all_sub_categories = set()
    for plot_data in data_array:
        all_sub_categories.update(plot_data.data.keys())

    all_sub_categories = sorted(
        list(all_sub_categories)
    )  # Sort for consistent ordering

    # Number of categories (bars) to plot
    num_categories = len(data_array)
    num_sub_categories = len(all_sub_categories)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up X-axis positions for each category
    bar_width = 0.7 / num_sub_categories  # Bar width for each sub-category
    category_positions = np.arange(
        num_categories
    )  # Positions of categories on the X-axis

    # Plot each sub-category in the same chart, offsetting their position to align with categories
    for i, sub_category in enumerate(all_sub_categories):
        sub_category_values = []

        # Collect the values for each sub-category, putting 0 if not present
        for plot_data in data_array:
            sub_category_values.append(plot_data.data.get(sub_category, 0))

        # Offset positions for each sub-category
        positions = category_positions + i * bar_width

        # Plot the sub-category as a bar
        ax.bar(positions, sub_category_values, bar_width, label=sub_category)

    # Set the X-axis labels (category names)
    ax.set_xticks(category_positions + (num_sub_categories - 1) * bar_width / 2)
    ax.set_xticklabels([plot_data.name for plot_data in data_array])

    # Add a legend to differentiate sub-categories
    ax.legend(title="Algorithms")

    # Add labels and title
    ax.set_xlabel("Mutation")
    ax.set_ylabel("Score (more = better)")
    ax.set_title("Algorithm resiliance to mutations relative to other images")

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # Example nested dictionary
    data_dict = {
        "Plot 1": {
            "Dataset A": np.random.normal(50, 10, 1000),
            "Dataset B": np.random.normal(60, 15, 1000),
        },
        "Plot 2": {
            "Dataset C": np.random.uniform(20, 80, 1000),
            "Dataset D": np.random.exponential(30, 1000),
        },
        "Plot 3": {
            "Dataset E": np.random.normal(40, 20, 1000),
        },
        "Plot 4": {
            "Dataset F": np.random.normal(70, 5, 1000),
            "Dataset G": np.random.normal(65, 7, 1000),
            "Dataset H": np.random.normal(75, 8, 1000),
        },
        "Plot 5": {
            "Dataset I": np.random.normal(55, 12, 1000),
            "Dataset J": np.random.normal(50, 10, 1000),
        },
    }

    # Call the function to plot and save the histograms
    plot_histograms(data_dict, filename="complex_histograms.png", num_bins=30)
