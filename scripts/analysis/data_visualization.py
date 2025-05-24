"""
Data Visualization Script for Time Series Analysis
This script provides visualization functions for time series data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_entities_over_time(data, save_path=None):
    """
    Plot observed entities over time.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with Year as index and ObservedEntities column
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=data.index, y='ObservedEntities', data=data, marker='o')
    plt.title('Observed Entities Over Time (1970-2024)')
    plt.xlabel('Year')
    plt.ylabel('Number of Observed Entities')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_entities_with_documents_histogram(data, save_path=None):
    """
    Plot observed entities over time with total documents histogram.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with Year as index, ObservedEntities and TotalDocuments columns
    save_path : str, optional
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    # Main plot: Entities over time
    ax1.plot(data.index, data['ObservedEntities'], marker='o', linewidth=2, markersize=6, label='Observed Entities')
    ax1.set_ylabel('Number of Observed Entities', fontsize=12)
    ax1.set_title('Observed Entities Over Time with Document Volume Context', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Histogram: Total documents distribution over time
    ax2.bar(data.index, data['TotalDocuments'], alpha=0.7, color='orange', width=0.8)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Total Documents', fontsize=12)
    ax2.set_title('Total Documents Volume by Year', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Format x-axis to show fewer ticks if too many years
    if len(data.index) > 20:
        step = max(1, len(data.index) // 10)
        ax1.set_xticks(data.index[::step])
        ax2.set_xticks(data.index[::step])
        ax1.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_entities_normalized_and_absolute(data, save_path=None):
    """
    Plot both absolute entities and entities normalized by total documents.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with Year as index, ObservedEntities and TotalDocuments columns
    save_path : str, optional
        Path to save the plot
    """
    # Calculate normalized entities (entities per document * 1000 for readability)
    data_with_normalized = data.copy()
    data_with_normalized['NormalizedEntities'] = (data['ObservedEntities'] / data['TotalDocuments']) * 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Top plot: Absolute entities with document histogram
    ax1_hist = ax1.twinx()

    # Line plot for entities
    ax1.plot(data.index, data['ObservedEntities'], marker='o', linewidth=2,
             markersize=6, color='blue', label='Observed Entities')
    ax1.set_ylabel('Number of Observed Entities', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Absolute Entities vs Document Volume', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Bar plot for documents
    ax1_hist.bar(data.index, data['TotalDocuments'], alpha=0.3, color='orange',
                 width=0.8, label='Total Documents')
    ax1_hist.set_ylabel('Total Documents', color='orange', fontsize=12)
    ax1_hist.tick_params(axis='y', labelcolor='orange')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_hist.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Bottom plot: Normalized entities
    ax2.plot(data.index, data_with_normalized['NormalizedEntities'], marker='s',
             linewidth=2, markersize=6, color='green', label='Entities per 1000 Documents')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Entities per 1000 Documents', color='green', fontsize=12)
    ax2.set_title('Document-Normalized Entity Trends', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.tick_params(axis='y', labelcolor='green')

    # Format x-axis
    if len(data.index) > 20:
        step = max(1, len(data.index) // 10)
        ax1.set_xticks(data.index[::step])
        ax2.set_xticks(data.index[::step])
        ax1.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return data_with_normalized


def plot_documents_over_time(data, save_path=None):
    """
    Plot total documents over time.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with Year as index and TotalDocuments column
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=data.index, y='TotalDocuments', data=data, marker='o', color='orange')
    plt.title('Total Documents Over Time (1970-2024)')
    plt.xlabel('Year')
    plt.ylabel('Total Number of Documents')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_entities_vs_documents(data, save_path=None):
    """
    Plot observed entities vs total documents with year as color.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with ObservedEntities and TotalDocuments columns
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='TotalDocuments',
        y='ObservedEntities',
        data=data,
        hue=data.index,
        palette='viridis',
        s=100
    )
    plt.title('Observed Entities vs. Total Documents (Color by Year)')
    plt.xlabel('Total Documents')
    plt.ylabel('Observed Entities')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_all_plots(data, output_dir=None):
    """
    Create all visualization plots for the data.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with the time series data
    output_dir : str, optional
        Directory to save plots
    """
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        plot_entities_over_time(data, f"{output_dir}/entities_over_time.png")
        plot_documents_over_time(data, f"{output_dir}/documents_over_time.png")
        plot_entities_vs_documents(data, f"{output_dir}/entities_vs_documents.png")
        plot_entities_with_documents_histogram(data, f"{output_dir}/entities_with_documents_histogram.png")
        normalized_data = plot_entities_normalized_and_absolute(data, f"{output_dir}/entities_normalized_and_absolute.png")
    else:
        plot_entities_over_time(data)
        plot_documents_over_time(data)
        plot_entities_vs_documents(data)
        plot_entities_with_documents_histogram(data)
        normalized_data = plot_entities_normalized_and_absolute(data)

    return normalized_data


if __name__ == "__main__":
    # Load data and create plots
    try:
        data = pd.read_csv('/home/callebalik/EnviDis/data/processed/sample_time_series_data.csv', index_col='Year')
        create_all_plots(data)
    except FileNotFoundError:
        print("Sample data not found. Please run data_generation.py first.")
