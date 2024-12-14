import flwr as fl
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense, Dropout
from tf_keras.optimizers import Adam
from utils import getDataset, evaluate_metrics, apply_noise

# Fetch dataset for Client 2
x_train, y_train, x_test, y_test = getDataset(client_id=2, num_clients=3)

# Create and compile model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, 30)),
    Dropout(0.5),
    Dense(65, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

class FlwrClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
        return model.get_weights(), len(x_train), {
            "loss": history.history["loss"][-1],  # Last epoch loss
            "accuracy": history.history["accuracy"][-1]  # Last epoch accuracy
        }

    def evaluate(self, parameters, config):
        model.set_weights(parameters)

        # Calculate predictions and metrics
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        y_pred_probs = model.predict(x_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        # Compute additional metrics
        _, recall, precision, f1 = evaluate_metrics(y_test, y_pred)
        noisy_pred = apply_noise(y_pred)
        noisy_accuracy, _, _, _ = evaluate_metrics(y_test, noisy_pred)

        # Prepare metrics dictionary
        metrics = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "noisy_accuracy": noisy_accuracy
        }

        # Return loss, number of examples, and metrics
        return loss, len(x_test), metrics

fl.client.start_numpy_client(server_address="localhost:8080", client=FlwrClient())