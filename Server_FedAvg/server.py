from typing import Dict, Optional, Tuple
import flwr as fl
from utils.utils import getDataset,get_model
from tf_keras.optimizers import Adam
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import getDataset, evaluate_metrics, apply_noise_iterative

model,lr,earlystop = get_model()

model.compile(
    optimizer=Adam(learning_rate=lr),  # Add the scheduler here
    loss='binary_crossentropy',
    metrics=['accuracy']
)

client_metrics = []
privacy_metrics = []

def aggregate_fit_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate client-reported metrics."""
    aggregated_metrics = {}
    for metric_name in metrics[0].keys():
        aggregated_metrics[metric_name] = sum(metric[metric_name] for metric in metrics) / len(metrics)
    return aggregated_metrics

def get_global_test_data(num_clients):
    x_test_list = []
    y_test_list = []
    for client_id in range(num_clients):
        _, _, x_test, y_test = getDataset(client_id, num_clients)
        x_test_list.append(x_test)
        y_test_list.append(y_test)
    return np.concatenate(x_test_list), np.concatenate(y_test_list)

def get_eval_fn(model, num_clients):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round, parameters, config):
        global client_metrics, privacy_metrics

        # Set model parameters
        model.set_weights(parameters)

        # Fetch test data for global evaluation (aggregated or client-specific)
        x_test, y_test = get_global_test_data(num_clients=num_clients)

        # Evaluate the model directly
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

        # Get predictions for detailed metrics
        y_pred_probs = model.predict(x_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        # Compute standard metrics
        accuracy, recall, precision, f1 = evaluate_metrics(y_test, y_pred)

        # Apply noise iteratively using predefined scales
        noise_scales = [0.1, 1, 2, 5, 10]  # Define noise scales
        noisy_predictions = apply_noise_iterative(y_pred, noise_scales=noise_scales)

        # Iterate through each noise scale and compute metrics
        for noise_scale, noisy_pred in noisy_predictions.items():
            noisy_accuracy, noisy_recall, noisy_precision, noisy_f1 = evaluate_metrics(y_test, noisy_pred)
            privacy_budget = 1 / noise_scale  # Example privacy budget calculation

            # Store privacy metrics for each scale
            privacy_metrics.append({
                "round": server_round,
                "noise_scale": noise_scale,
                "privacy_budget": privacy_budget,
                "noisy_accuracy": noisy_accuracy,
                "noisy_recall": noisy_recall,
                "noisy_precision": noisy_precision,
                "noisy_f1": noisy_f1,
            })

            # Log metrics for this noise scale
            print(f"Noise Scale: {noise_scale:.1f}")
            print(f"Noisy Accuracy: {noisy_accuracy:.4f}, Noisy Recall: {noisy_recall:.4f}, "
                  f"Noisy Precision: {noisy_precision:.4f}, Noisy F1: {noisy_f1:.4f}, Privacy Budget: {privacy_budget:.4f}")

        # Store standard metrics for visualization
        client_metrics.append({
            "round": server_round,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1
        })

        # Log standard metrics
        print(f"Round {server_round} Metrics:")
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")

        # Return metrics to the federated server
        return loss, {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

    return evaluate

# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_fn=get_eval_fn(model,2),
)

# Start server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
)