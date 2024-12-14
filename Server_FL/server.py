from typing import Dict, Optional, Tuple
import flwr as fl
from utils import model, getDataset
from tf_keras.optimizers import Adam
from typing import Dict, List
import matplotlib.pyplot as plt
from utils import getDataset, evaluate_metrics, apply_noise

model.compile(
    optimizer=Adam(learning_rate=0.001),  
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

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round, parameters, config):
        global client_metrics, privacy_metrics
        model.set_weights(parameters)

        # Fetch test data for global evaluation
        x_test, y_test = getDataset(client_id=0)[2:]  # Assume client 0's data for global evaluation

        # Get predictions and evaluate standard metrics
        y_pred_probs = model.predict(x_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        accuracy, recall, precision, f1 = evaluate_metrics(y_test, y_pred)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

        # Calculate noise-related metrics
        noise_scale = 0.1  # Example noise scale
        noisy_pred = apply_noise(y_pred, noise_scale)
        noisy_accuracy, _, _, _ = evaluate_metrics(y_test, noisy_pred)
        privacy_budget = 1 / noise_scale

        # Store metrics for visualization
        client_metrics.append({
            "round": server_round,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1
        })
        privacy_metrics.append({
            "round": server_round,
            "noise_scale": noise_scale,
            "privacy_budget": privacy_budget,
            "noisy_accuracy": noisy_accuracy
        })

        # Log metrics for debugging
        print(f"Round {server_round} Metrics:")
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, Noisy Accuracy: {noisy_accuracy:.4f}, Privacy Budget: {privacy_budget:.4f}")

        return loss, {
            "noisy scale" : noise_scale,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "noisy_accuracy": noisy_accuracy,
            "privacy_budget": privacy_budget,
        }
    return evaluate

# Visualization of Metrics
def plot_metrics(client_metrics):
    """Plot standard metrics over federated rounds."""
    rounds = [m["round"] for m in client_metrics]
    accuracies = [m["accuracy"] for m in client_metrics]
    recalls = [m["recall"] for m in client_metrics]
    precisions = [m["precision"] for m in client_metrics]
    f1_scores = [m["f1"] for m in client_metrics]

    metrics = {
        "Accuracy": accuracies,
        "Recall": recalls,
        "Precision": precisions,
        "F1 Score": f1_scores
    }

    for metric_name, values in metrics.items():
        plt.plot(rounds, values, marker='o', label=metric_name)

    plt.xlabel("Rounds")
    plt.ylabel("Metric Value")
    plt.title("Metrics Over Federated Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_privacy_metrics(privacy_metrics):
    """Plot noise, privacy budget, and accuracy after noise."""
    rounds = [m["round"] for m in privacy_metrics]
    noise_scales = [m["noise_scale"] for m in privacy_metrics]
    privacy_budgets = [m["privacy_budget"] for m in privacy_metrics]
    noisy_accuracies = [m["noisy_accuracy"] for m in privacy_metrics]

    # Noise scale
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, noise_scales, marker='o', label="Noise Scale", color="blue")
    plt.xlabel("Rounds")
    plt.ylabel("Noise Scale")
    plt.title("Noise Scale Over Federated Rounds")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Privacy budget
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, privacy_budgets, marker='o', label="Privacy Budget", color="green")
    plt.xlabel("Rounds")
    plt.ylabel("Privacy Budget")
    plt.title("Privacy Budget Over Federated Rounds")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Accuracy after noise
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, noisy_accuracies, marker='o', label="Noisy Accuracy", color="red")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy After Noise")
    plt.title("Accuracy After Noise Over Federated Rounds")
    plt.grid(True)
    plt.legend()
    plt.show()

# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_fn=get_eval_fn(model),
)

# Start server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
)

# After training, plot the metrics
plot_metrics(client_metrics)
plot_privacy_metrics(privacy_metrics)