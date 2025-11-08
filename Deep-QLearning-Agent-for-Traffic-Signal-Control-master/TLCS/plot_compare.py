import os
import numpy as np
import matplotlib.pyplot as plt
import json

def load_data(model_path, metric):
    """
    Load data from the plot_{metric}_data.txt file
    
    Args:
        model_path: Path to the model folder (SUMO or LSTM)
        metric: One of 'reward', 'delay', or 'queue'
    
    Returns:
        List of values for the given metric
    """
    data_path = os.path.join(model_path, f'plot_{metric}_data.txt')
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found!")
        return []
    
    with open(data_path, 'r') as f:
        # Each line contains one value
        return [float(line.strip()) for line in f.readlines()]

def create_comparison_plot(sumo_data, lstm_data, metric, output_path, model_version):
    """
    Create a comparison plot for a specific metric
    
    Args:
        sumo_data: List of values for SUMO model
        lstm_data: List of values for LSTM model
        metric: One of 'reward', 'delay', or 'queue'
        output_path: Path to save the output plots
        model_version: Model version number for the output filename
    """
    # Define labels and titles based on metric
    if metric == 'reward':
        ylabel = 'Cumulative Negative Reward'
        title = 'Reward Comparison: SUMO vs LSTM'
    elif metric == 'delay':
        ylabel = 'Cumulative Delay (s)'
        title = 'Delay Comparison: SUMO vs LSTM'
    elif metric == 'queue':
        ylabel = 'Average Queue Length (vehicles)'
        title = 'Queue Length Comparison: SUMO vs LSTM'
    else:
        ylabel = metric
        title = f'{metric.capitalize()} Comparison: SUMO vs LSTM'
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot data
    episodes = range(1, min(len(sumo_data) + 1, len(lstm_data) + 1))
    
    # Truncate data to the same length
    min_len = min(len(sumo_data), len(lstm_data))
    sumo_data = sumo_data[:min_len]
    lstm_data = lstm_data[:min_len]
    
    # Plot SUMO data
    plt.plot(
        episodes, 
        sumo_data, 
        color='blue', 
        linestyle='-', 
        linewidth=2,
        label='SUMO'
    )
    
    # Plot LSTM data
    plt.plot(
        episodes, 
        lstm_data, 
        color='red', 
        linestyle='--', 
        linewidth=2,
        label='LSTM'
    )
    
    # Calculate statistics
    if len(sumo_data) > 0 and len(lstm_data) > 0:
        sumo_avg = sum(sumo_data) / len(sumo_data)
        lstm_avg = sum(lstm_data) / len(lstm_data)
        diff_pct = abs((lstm_avg - sumo_avg) / sumo_avg * 100) if sumo_avg != 0 else 0
        
        textstr = f'Avg SUMO: {sumo_avg:.2f}\nAvg LSTM: {lstm_avg:.2f}\nDiff: {diff_pct:.2f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
    
    # Add labels and styling
    plt.title(title, fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    filename = f"compare_{metric}_model{model_version}.png"
    plt.savefig(os.path.join(output_path, filename), dpi=300)
    plt.close()
    
    print(f"Saved {filename}")
    
    # Calculate and return error metrics
    error_metrics = {}
    if len(sumo_data) > 0 and len(lstm_data) > 0:
        # Mean Absolute Error
        mae = sum(abs(s - l) for s, l in zip(sumo_data, lstm_data)) / len(sumo_data)
        
        # Mean Squared Error
        mse = sum((s - l)**2 for s, l in zip(sumo_data, lstm_data)) / len(sumo_data)
        
        # Root Mean Squared Error
        rmse = mse ** 0.5
        
        # Mean Percentage Error (avoiding division by zero)
        percentage_errors = []
        for s, l in zip(sumo_data, lstm_data):
            if s != 0:
                percentage_errors.append(abs((s - l) / s) * 100)
        
        mpe = sum(percentage_errors) / len(percentage_errors) if percentage_errors else 0

        # Calculate MAPE (Mean Absolute Percentage Error)
        absolute_percentage_errors = []
        for s, l in zip(sumo_data, lstm_data):
            if s != 0:  # Avoid division by zero
                absolute_percentage_errors.append(abs((s - l) / s) * 100)
        
        mape = sum(absolute_percentage_errors) / len(absolute_percentage_errors) if absolute_percentage_errors else 0
        
        error_metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MPE': mpe,
            'MAPE': mape
        }
    
    return error_metrics

def calculate_lstm_performance_gain(sumo_data, lstm_data):
    """
    Calculate how much better or worse LSTM performs compared to SUMO
    """
    if not sumo_data or not lstm_data:
        return None
    
    # Truncate to same length
    min_len = min(len(sumo_data), len(lstm_data))
    sumo_data = sumo_data[:min_len]
    lstm_data = lstm_data[:min_len]
    
    # For reward and delay metrics, lower is better
    # For queue length, lower is better too
    # So for all metrics, calculate: (sumo - lstm) / sumo * 100
    # Positive value means LSTM is better, negative means SUMO is better
    
    sumo_avg = sum(sumo_data) / len(sumo_data)
    lstm_avg = sum(lstm_data) / len(lstm_data)
    
    if sumo_avg == 0:
        return 0  # Avoid division by zero
    
    gain = (sumo_avg - lstm_avg) / sumo_avg * 100
    return gain


def main():
    # Detect the model version from the folder structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    model_folders = [folder for folder in os.listdir(models_dir) 
                    if folder.startswith('model_') and os.path.isdir(os.path.join(models_dir, folder))]
    if not model_folders:
        print("No model folders found. Make sure the script is in the same directory as the model folders.")
        return
    
    # Sort model folders to get the latest one
    model_folders.sort()
    latest_model = model_folders[-1]
    model_version = latest_model.split('_')[1]
    
    print(f"Processing model version: {model_version}")
    
    # Define paths
    model_path = os.path.join(models_dir, latest_model)
    sumo_path = os.path.join(model_path, 'SUMO')
    lstm_path = os.path.join(model_path, 'LSTM')
    output_path = os.path.join(models_dir, 'comparison_plots')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Define metrics to compare
    metrics = ['reward', 'delay', 'queue']
    all_error_metrics = {}
    performance_gains = {}
    
    # Process each metric
    for metric in metrics:
        print(f"\nProcessing {metric} data...")
        
        # Load data
        sumo_data = load_data(sumo_path, metric)
        lstm_data = load_data(lstm_path, metric)
        
        if not sumo_data or not lstm_data:
            print(f"Missing data for {metric}, skipping...")
            continue
        
        # Create comparison plot and get error metrics
        error_metrics = create_comparison_plot(sumo_data, lstm_data, metric, output_path, model_version)
        all_error_metrics[metric] = error_metrics
        
        # Calculate performance gain
        gain = calculate_lstm_performance_gain(sumo_data, lstm_data)
        performance_gains[metric] = gain
        
        print(f"Performance gain for {metric}: {gain:.2f}% ({'better' if gain > 0 else 'worse'})")
    
    # Save error metrics to a file
    with open(os.path.join(output_path, f'error_metrics_model{model_version}.txt'), 'w') as f:
        for metric, errors in all_error_metrics.items():
            f.write(f"=== {metric.upper()} ===\n")
            for name, value in errors.items():
                f.write(f"{name}: {value:.4f}\n")
            f.write("\n")
    
    # Save performance gains to a file
    with open(os.path.join(output_path, f'performance_gains_model{model_version}.txt'), 'w') as f:
        for metric, gain in performance_gains.items():
            f.write(f"{metric}: {gain:.2f}% ({'better' if gain > 0 else 'worse'})\n")
    
    print(f"\nAll comparison plots and metrics saved to {output_path}")

if __name__ == "__main__":
    main()