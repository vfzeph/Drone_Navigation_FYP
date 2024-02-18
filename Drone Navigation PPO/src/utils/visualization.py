import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Example of a dynamic updating plot for Matplotlib
def dynamic_plot_example(update_function, frames=100, interval=100):
    """
    A dynamic plot that updates in real-time.
    Args:
        update_function: The function to call at each frame to update the plot.
        frames: The total number of frames (updates).
        interval: Time between updates in milliseconds.
    """
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update_function, frames=range(frames), interval=interval)
    plt.show()

# Advanced customization for static Matplotlib plot
def enhanced_plot_with_customization(metrics):
    """
    A static plot with advanced customization features.
    Args:
        metrics: A dictionary of metric names to lists of values.
    """
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    for metric_name, values in metrics.items():
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker='o', linewidth=2, label=metric_name)
        
        # Highlighting the last value
        ax.annotate(f"{values[-1]:.2f}", xy=(len(values), values[-1]), xytext=(5,5), 
                    textcoords="offset points", weight='bold', color='darkslategray')

    ax.set(title="Enhanced Training Progress", xlabel="Epoch", ylabel="Metric Value")
    ax.legend(frameon=True, loc='upper left', fontsize='small')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# Incorporating the improvements
if __name__ == "__main__":
    # Dynamic Plotting Example (placeholder for the update function)
    def update(frame):
        # This function would update the plot elements for each frame
        pass

    # dynamic_plot_example(update_function=update)
    
    # Advanced Customization Example
    metrics = {'Accuracy': np.random.rand(10).cumsum(), 'Loss': np.random.rand(10).cumsum()}
    enhanced_plot_with_customization(metrics)
