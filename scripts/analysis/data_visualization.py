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
    else:
        plot_entities_over_time(data)
        plot_documents_over_time(data)
        plot_entities_vs_documents(data)


if __name__ == "__main__":
    # Load data and create plots
    try:
        data = pd.read_csv('/home/callebalik/EnviDis/data/processed/sample_time_series_data.csv', index_col='Year')
        create_all_plots(data)
    except FileNotFoundError:
        print("Sample data not found. Please run data_generation.py first.")
