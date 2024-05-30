import matplotlib.pyplot as plt
import seaborn as sns
import logging
import plotly.express as px

class DataVisualizer:
    """
    A class to handle various types of data visualizations, with integrated logging for monitoring visualization processes.
    Uses Seaborn, Matplotlib, and Plotly for plotting.
    """

    def __init__(self, logger=None):
        # Setting up the seaborn style for all plots
        sns.set(style="whitegrid")
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("DataVisualizer initialized with Seaborn style set to 'whitegrid'.")

    def plot_time_series(self, data, x, y, title="Time Series Plot", xlabel="Time", ylabel="Value", save_path=None):
        """
        Generate a time series plot with logging of key steps.

        Args:
            data (DataFrame): Pandas DataFrame containing the data to plot.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data, x=x, y=y)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Time Series plot saved to %s.", save_path)
            plt.show()
            self.logger.info("Time Series plot created successfully for %s vs %s.", x, y)
        except Exception as e:
            self.logger.error("Failed to create time series plot: %s", e)

    def plot_histogram(self, data, column, title="Histogram", bins=30, save_path=None):
        """
        Generate a histogram for a specified column with error handling and logging.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            column (str): Column to plot histogram for.
            title (str): Title of the histogram.
            bins (int): Number of bins for the histogram.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[column], bins=bins, kde=True)
            plt.title(title)
            plt.xlabel(column)
            plt.ylabel("Frequency")
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Histogram saved to %s.", save_path)
            plt.show()
            self.logger.info("Histogram plotted successfully for column: %s.", column)
        except Exception as e:
            self.logger.error("Failed to plot histogram for %s: %s", column, e)

    def plot_correlation_matrix(self, data, title="Correlation Matrix", save_path=None):
        """
        Generate a heatmap for the correlation matrix of the dataframe.
        
        Args:
            data (DataFrame): Pandas DataFrame to compute the correlation matrix from.
            title (str): Title for the correlation matrix plot.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(10, 8))
            corr = data.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title(title)
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Correlation matrix heatmap saved to %s.", save_path)
            plt.show()
            self.logger.info("Correlation matrix heatmap generated successfully.")
        except Exception as e:
            self.logger.error("Failed to generate correlation matrix heatmap: %s", e)

    def plot_scatter(self, data, x, y, hue=None, title="Scatter Plot", xlabel="X", ylabel="Y", save_path=None):
        """
        Generate a scatter plot with detailed logging.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            hue (str, optional): Column name for grouping variable that will produce points with different colors.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x=x, y=y, hue=hue)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Scatter plot saved to %s.", save_path)
            plt.show()
            self.logger.info("Scatter plot created successfully for %s vs %s.", x, y)
        except Exception as e:
            self.logger.error("Failed to create scatter plot for %s vs %s: %s", x, y, e)

    def plot_pair(self, data, hue=None, title="Pair Plot", save_path=None):
        """
        Generate a pair plot for the dataframe.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            hue (str, optional): Column name for grouping variable that will produce points with different colors.
            title (str): Title for the pair plot.
            save_path (str): Optional path to save the plot image.
        """
        try:
            pair_plot = sns.pairplot(data, hue=hue)
            pair_plot.fig.suptitle(title, y=1.02)
            if save_path:
                pair_plot.savefig(save_path)
                self.logger.info("Pair plot saved to %s.", save_path)
            plt.show()
            self.logger.info("Pair plot created successfully.")
        except Exception as e:
            self.logger.error("Failed to create pair plot: %s", e)

    def plot_distribution(self, data, column, title="Distribution Plot", save_path=None):
        """
        Generate a distribution plot for a specified column.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            column (str): Column to plot distribution for.
            title (str): Title of the distribution plot.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data[column], shade=True)
            plt.title(title)
            plt.xlabel(column)
            plt.ylabel("Density")
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Distribution plot saved to %s.", save_path)
            plt.show()
            self.logger.info("Distribution plot created successfully for column: %s.", column)
        except Exception as e:
            self.logger.error("Failed to create distribution plot for %s: %s", column, e)

    def interactive_scatter_plot(self, data, x, y, title="Interactive Scatter Plot"):
        """
        Generate an interactive scatter plot using Plotly.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            title (str): Title of the plot.
        """
        try:
            fig = px.scatter(data, x=x, y=y, title=title)
            fig.show()
            self.logger.info("Interactive scatter plot created successfully for %s vs %s.", x, y)
        except Exception as e:
            self.logger.error("Failed to create interactive scatter plot for %s vs %s: %s", x, y, e)
