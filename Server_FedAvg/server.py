import flwr as fl
from tf_keras.optimizers import Adam
from utils.utils import getDataset, get_model, evaluate_metrics
import numpy as np
import pandas as pd

# Load and compile the model
model, lr_schedule, earlystop = get_model()
model.compile(optimizer=Adam(learning_rate=lr_schedule),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Store metrics across rounds
evaluation_metrics = []

def get_global_test_data(num_clients):
    """Combine test data from all clients."""
    x_test_list, y_test_list = [], []
    for client_id in range(num_clients):
        _, _, x_test, y_test = getDataset(client_id, num_clients)
        x_test_list.append(x_test)
        y_test_list.append(y_test)
    return np.concatenate(x_test_list), np.concatenate(y_test_list)

def get_eval_fn(model, num_clients):
    """Server-side evaluation function."""
    def evaluate(server_round, parameters, config):
        global evaluation_metrics
        model.set_weights(parameters)
        
        # Aggregate global test data
        x_test, y_test = get_global_test_data(num_clients)

        # Evaluate clean data
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        y_pred_probs = model.predict(x_test)
        y_pred = (y_pred_probs > 0.5).astype(int)
        acc, recall, precision, f1 = evaluate_metrics(y_test, y_pred)

        # Apply noise iteratively and calculate privacy budget
        noise_scales = [0.1, 1, 2, 5, 10]
        delta = 1e-5
        sensitivity = 1.0  # Fixed sensitivity for predictions

        noisy_metrics = {}
        for noise_scale in noise_scales:
            noise = np.random.normal(0, noise_scale, y_pred.shape)
            noisy_pred = (y_pred + noise > 0.5).astype(int)
            noisy_acc, _, _, _ = evaluate_metrics(y_test, noisy_pred)

            # Calculate privacy budget (epsilon)
            epsilon = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / noise_scale

            noisy_metrics[f"noisy_accuracy_{noise_scale}"] = noisy_acc
            noisy_metrics[f"privacy_budget_{noise_scale}"] = epsilon

        # Store all metrics
        round_metrics = {
            "round": server_round,
            "loss": loss,
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            **noisy_metrics  # Include noisy metrics
        }
        evaluation_metrics.append(round_metrics)

        # Log metrics
        print(f"Round {server_round} Metrics:")
        print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
        for noise_scale in noise_scales:
            print(f"Noise Scale {noise_scale}, Noisy Accuracy: {noisy_metrics[f'noisy_accuracy_{noise_scale}']:.4f}, "
                  f"Privacy Budget: {noisy_metrics[f'privacy_budget_{noise_scale}']:.4f}")
        
        return loss, {"accuracy": acc, "f1": f1,"recall":recall,"precision":precision, **noisy_metrics}
    
    return evaluate

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_fn=get_eval_fn(model, num_clients=2))

# Save metrics to CSV after training
def save_metrics_to_csv(metrics, filename="evaluation_metrics.csv"):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")

# Start the FL server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
)

# Export metrics after training
save_metrics_to_csv(evaluation_metrics)