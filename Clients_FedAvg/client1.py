import flwr as fl
from tf_keras.optimizers import Adam
import pandas as pd
from utils.utils import getDataset, evaluate_metrics, apply_noise_iterative,get_model

# Fetch dataset for Client 1
x_train, y_train, x_test, y_test = getDataset(client_id=0, num_clients=2)
y_series = pd.Series(y_train.flatten())
counts = y_series.value_counts()
print(counts)

# Create and compile model
model,lr,earlystop = get_model()

model.compile(
    optimizer=Adam(learning_rate=lr),  # Add the scheduler here
    loss='binary_crossentropy',
    metrics=['accuracy']
)

class FlwrClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=32,
            validation_data=(x_test, y_test),
            callbacks=[earlystop],
            verbose=1
        )
        return model.get_weights(), len(x_train), {
            "loss": history.history["loss"][-1],  # Last epoch loss
            "accuracy": history.history["accuracy"][-1]  # Last epoch accuracy
        }

    def evaluate(self, parameters, config):
        model.set_weights(parameters)

        # Evaluate on the clean data
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

        # Generate predictions
        y_pred_probs = model.predict(x_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        # Compute standard metrics
        _, recall, precision, f1 = evaluate_metrics(y_test, y_pred)

        # Apply noise iteratively using predefined scales
        noise_scales = [0.1, 1, 2, 5, 10]
        noisy_predictions = apply_noise_iterative(y_pred, noise_scales=noise_scales)

        # Flatten noisy metrics into scalar entries
        noisy_metrics_flat = {}
        for noise_scale, noisy_pred in noisy_predictions.items():
            noisy_accuracy, noisy_recall, noisy_precision, noisy_f1 = evaluate_metrics(y_test, noisy_pred)
            noisy_metrics_flat[f"noisy_accuracy_{noise_scale}"] = noisy_accuracy
            noisy_metrics_flat[f"noisy_recall_{noise_scale}"] = noisy_recall
            noisy_metrics_flat[f"noisy_precision_{noise_scale}"] = noisy_precision
            noisy_metrics_flat[f"noisy_f1_{noise_scale}"] = noisy_f1

        # Log metrics for debugging
        print(f"Evaluation Metrics:")
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
        for noise_scale in noise_scales:
            print(f"Noise Scale {noise_scale}: Noisy Accuracy: {noisy_metrics_flat[f'noisy_accuracy_{noise_scale}']:.4f}, "
                  f"Noisy Recall: {noisy_metrics_flat[f'noisy_recall_{noise_scale}']:.4f}, "
                  f"Noisy Precision: {noisy_metrics_flat[f'noisy_precision_{noise_scale}']:.4f}, "
                  f"Noisy F1: {noisy_metrics_flat[f'noisy_f1_{noise_scale}']:.4f}")

        # Prepare metrics dictionary
        metrics = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            **noisy_metrics_flat  # Add noisy metrics as individual scalar entries
        }

        # Return loss, number of examples, and metrics
        return loss, len(x_test), metrics

fl.client.start_numpy_client(server_address="localhost:8080", client=FlwrClient())