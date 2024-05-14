import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Import seaborn for styling
import os  # For handling path operations

# Function to create enhanced matplotlib reports
def enhanced_matplotlib_report(data, save_path=None):
    """
    Generate a static plot with advanced customization features.
    
    Args:
        data (dict): Dictionary of metric names to their list of values.
        save_path (str): Optional path to save the plot as an image file.
    """
    sns.set(style='darkgrid')  # Use seaborn to apply the 'darkgrid' style
    fig, ax = plt.subplots(figsize=(12, 7))

    for metric_name, values in data.items():
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker='o', linewidth=2, label=metric_name)
        ax.annotate(f"{values[-1]:.2f}", xy=(len(values), values[-1]), xytext=(5, 5),
                    textcoords="offset points", weight='bold', color='darkslategray')

    ax.set(title="Enhanced Training Progress", xlabel="Epoch", ylabel="Metric Value")
    ax.legend(frameon=True, loc='upper left', fontsize='small')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
        plt.savefig(save_path, format='png', dpi=300)
    plt.show()

# Example usage within a simulation or model evaluation script
if __name__ == "__main__":
    # Example data that you might be plotting
    data = {
        'Accuracy': np.random.rand(10).cumsum(),  # Simulated accuracy metrics
        'Loss': np.random.rand(10).cumsum()  # Simulated loss metrics
    }

    # Calling the enhanced plotting function
    enhanced_matplotlib_report(data, save_path='path/to/save/your/plot.png')
