from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
import matplotlib.pyplot as plt
from lstm_model import LSTMPredictor
from training_simulation import Simulation
from generator import TrafficGenerator
from shutil import copyfile, rmtree
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
else:
    print("No GPU detected, running on CPU.")
    
def train_model(model_type, config, sumo_cmd, base_path):
    """Train a single model type and return its metrics"""
    print(f"\n=== Training {model_type} model ===")
    
    # Create model-specific paths
    model_path = os.path.join(base_path, model_type)
    os.makedirs(model_path, exist_ok=True)
    
    # Re-initialize components for clean training
    lstm_predictor = LSTMPredictor(config['num_states']) if model_type == "LSTM" else None
    
    model = TrainModel(
        config['num_layers'],
        config['width_layers'],
        config['batch_size'],
        config['learning_rate'],
        input_dim=config['num_states'],
        output_dim=config['num_actions']
    )
    
    traffic_gen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )
    
    memory = Memory(
        config['memory_size_max'],
        config['memory_size_min']
    )
    
    simulation = Simulation(
        model,
        memory,
        traffic_gen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        lstm_predictor
    )
    
    # Add pre-training for LSTM model
    if model_type == "LSTM":
        simulation.pre_train_lstm(episodes=3)  # Pre-train on 3 episodes
    
    # Train the model
    episode = 0
    while episode < config['total_episodes']:
        epsilon = 1.0 - (episode / config['total_episodes'])
        simulation.run(episode, epsilon, model_type=model_type)
        episode += 1
    
    # Save model and metrics
    model.save_model(model_path)
    
    # For LSTM model, also save the trained LSTM predictor
    if model_type == "LSTM" and lstm_predictor:
        lstm_predictor.model.save(os.path.join(model_path, "lstm_predictor.h5"))
        
        # Evaluate and save LSTM accuracy metrics
        accuracy_metrics = simulation.evaluate_lstm_accuracy()
        with open(os.path.join(model_path, "lstm_accuracy.txt"), "w") as f:
            for metric, value in accuracy_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
                print(f"LSTM {metric}: {value:.4f}")
    
    # Save individual model plots
    visualization = Visualization(model_path, dpi=96)
    visualization.save_data_and_plot(simulation.reward_store[model_type], 'reward', 'Episode', 'Cumulative negative reward')
    visualization.save_data_and_plot(simulation.cumulative_wait_store[model_type], 'delay', 'Episode', 'Cumulative delay (s)')
    visualization.save_data_and_plot(simulation.avg_queue_length_store[model_type], 'queue', 'Episode', 'Average queue length (vehicles)')
    
    return {
        'reward': simulation.reward_store,
        'delay': simulation.cumulative_wait_store,
        'queue': simulation.avg_queue_length_store
    }

    
def create_comparative_plots(metrics, base_path):
    """
    Creates three separate graphs comparing LSTM and SUMO for each metric:
    reward, queue length, and delay.
    
    Args:
        metrics: Dictionary containing metrics for both models
        base_path: Path to save the plots
    """
    # Define plot configurations
    plot_configs = [
        {
            'metric': 'reward',
            'ylabel': 'Cumulative Negative Reward',
            'title': 'Reward Comparison: SUMO vs LSTM',
            'filename': 'reward_comparison.png'
        },
        {
            'metric': 'queue',
            'ylabel': 'Average Queue Length (vehicles)',
            'title': 'Queue Length Comparison: SUMO vs LSTM',
            'filename': 'queue_comparison.png'
        },
        {
            'metric': 'delay',
            'ylabel': 'Cumulative Delay (s)',
            'title': 'Delay Comparison: SUMO vs LSTM',
            'filename': 'delay_comparison.png'
        }
    ]
    
    # Create each plot
    for config in plot_configs:
        plt.figure(figsize=(12, 6))
        
        # Get actual data lists to plot
        sumo_data = metrics['reward'] if config['metric'] == 'reward' else metrics[config['metric']]
        lstm_data = metrics['reward'] if config['metric'] == 'reward' else metrics[config['metric']]
        
        # Plot SUMO data
        plt.plot(
            sumo_data['SUMO'], 
            color='blue', 
            linestyle='-', 
            linewidth=2,
            label='SUMO'
        )
        
        # Plot LSTM data
        plt.plot(
            lstm_data['LSTM'], 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label='LSTM'
        )
        
        # Add labels and styling
        plt.title(config['title'], fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel(config['ylabel'], fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Calculate some statistics for the text box
        if len(sumo_data['SUMO']) > 0 and len(lstm_data['LSTM']) > 0:
            sumo_avg = sum(sumo_data['SUMO']) / len(sumo_data['SUMO'])
            lstm_avg = sum(lstm_data['LSTM']) / len(lstm_data['LSTM'])
            diff_pct = abs((lstm_avg - sumo_avg) / sumo_avg * 100) if sumo_avg != 0 else 0
            
            textstr = f'Avg SUMO: {sumo_avg:.2f}\nAvg LSTM: {lstm_avg:.2f}\nDiff: {diff_pct:.2f}%'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', bbox=props)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, config['filename']), dpi=300)
        plt.close()
    
    print(f"Comparative plots saved to {base_path}")

    

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    base_path = set_train_path(config['models_path_name'])

    
    # Clean data directories
    for folder in ['csv', 'lstm_csv']:
        if os.path.exists(folder):
            rmtree(folder)
        os.makedirs(folder)
    
     # Train both models and collect metrics
    sumo_metrics = train_model("SUMO", config, sumo_cmd, base_path)
    lstm_metrics = train_model("LSTM", config, sumo_cmd, base_path)
    
    # Initialize metrics structure with actual data
    metrics = {
    'reward': {'SUMO': sumo_metrics['reward']['SUMO'], 'LSTM': lstm_metrics['reward']['LSTM']},
    'delay': {'SUMO': sumo_metrics['delay']['SUMO'], 'LSTM': lstm_metrics['delay']['LSTM']},
    'queue': {'SUMO': sumo_metrics['queue']['SUMO'], 'LSTM': lstm_metrics['queue']['LSTM']}
    }

    # create_comparative_plots(metrics, base_path)
    
    # # Create comparative plots
    # def create_comparison(metric, ylabel):
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(metrics['SUMO'][metric]['SUMO'], label='SUMO Actual')
    #     plt.plot(metrics['LSTM'][metric]['LSTM'], label='LSTM Predicted')
    #     plt.title(f"{ylabel} Comparison")
    #     plt.xlabel("Episode")
    #     plt.ylabel(ylabel)
    #     plt.legend()
    #     plt.savefig(os.path.join(base_path, f'{metric}_comparison.png'))
    #     plt.close()

    # def calculate_and_plot_loss(metrics, base_path):
    #     """
    #     Calculate the loss between SUMO and LSTM predictions and plot the results.
    #     """
    #     # Get the rewards from both models
    #     sumo_rewards = metrics['SUMO']['reward']['SUMO']
    #     lstm_rewards = metrics['LSTM']['reward']['LSTM']
        
    #     # Ensure both lists are the same length
    #     min_length = min(len(sumo_rewards), len(lstm_rewards))
    #     sumo_rewards = sumo_rewards[:min_length]
    #     lstm_rewards = lstm_rewards[:min_length]
        
    #     # Calculate various loss metrics
    #     absolute_differences = [abs(s - l) for s, l in zip(sumo_rewards, lstm_rewards)]
    #     mean_absolute_error = sum(absolute_differences) / min_length
        
    #     squared_differences = [(s - l)**2 for s, l in zip(sumo_rewards, lstm_rewards)]
    #     mean_squared_error = sum(squared_differences) / min_length
    #     root_mean_squared_error = mean_squared_error ** 0.5
        
    #     # Calculate percentage error (avoid division by zero)
    #     percentage_errors = []
    #     for s, l in zip(sumo_rewards, lstm_rewards):
    #         if s != 0:
    #             percentage_errors.append(abs((s - l) / s) * 100)
    #         else:
    #             percentage_errors.append(0)  # Skip or set to 0 when actual is 0


        
        
    #     mean_percentage_error = sum(percentage_errors) / len(percentage_errors) if percentage_errors else 0
        
    #     # Plot the absolute differences
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(absolute_differences, label='Absolute Difference')
    #     plt.axhline(y=mean_absolute_error, color='r', linestyle='--', label=f'Mean Absolute Error: {mean_absolute_error:.2f}')
    #     plt.title('Absolute Difference Between SUMO and LSTM Rewards')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Absolute Difference')
    #     plt.legend()
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.savefig(os.path.join(base_path, 'reward_absolute_difference.png'))
    #     plt.close()
        
    #     # Plot the percentage errors
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(percentage_errors, label='Percentage Error')
    #     plt.axhline(y=mean_percentage_error, color='r', linestyle='--', label=f'Mean Percentage Error: {mean_percentage_error:.2f}%')
    #     plt.title('Percentage Error Between SUMO and LSTM Rewards')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Percentage Error (%)')
    #     plt.legend()
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.savefig(os.path.join(base_path, 'reward_percentage_error.png'))
    #     plt.close()
        
    #     # Save the error metrics to a file
    #     with open(os.path.join(base_path, 'error_metrics.txt'), 'w') as f:
    #         f.write(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f}\n")
    #         f.write(f"Mean Squared Error (MSE): {mean_squared_error:.4f}\n")
    #         f.write(f"Root Mean Squared Error (RMSE): {root_mean_squared_error:.4f}\n")
    #         f.write(f"Mean Percentage Error: {mean_percentage_error:.4f}%\n")
        
    #     # Return the calculated metrics
    #     return {
    #         'MAE': mean_absolute_error,
    #         'MSE': mean_squared_error,
    #         'RMSE': root_mean_squared_error,
    #         'MPE': mean_percentage_error
    #     }
        
    # create_comparison('reward', 'Cumulative Negative Reward')
    # create_comparison('delay', 'Cumulative Delay (s)')
    # create_comparison('queue', 'Average Queue Length (vehicles)')
    
    # Save config file
    copyfile('training_settings.ini', os.path.join(base_path, 'training_settings.ini'))
    
    print("\nTraining complete!")
    print(f"All results stored in: {base_path}")
    print(f"Individual model results in: {os.path.join(base_path, 'SUMO')} and {os.path.join(base_path, 'LSTM')}")
    print(f"Comparative plots saved directly in: {base_path}")

    # After creating the comparison plots
    # error_metrics = calculate_and_plot_loss(metrics, base_path)
    # print("\nError Metrics:")
    # print(f"Mean Absolute Error (MAE): {error_metrics['MAE']:.4f}")
    # print(f"Root Mean Squared Error (RMSE): {error_metrics['RMSE']:.4f}")
    # print(f"Mean Percentage Error: {error_metrics['MPE']:.4f}%")

