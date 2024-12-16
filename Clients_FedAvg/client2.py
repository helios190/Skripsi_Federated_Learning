import flwr as fl
from tf_keras.optimizers import Adam
from utils.utils import getDataset, get_model, evaluate_metrics

# Fetch dataset for Client 1
x_train, y_train, x_test, y_test = getDataset(client_id=1, num_clients=2,split_ratios=[0.6,0.4])

# Load and compile model
model, lr_schedule, earlystop = get_model()
model.compile(optimizer=Adam(learning_rate=lr_schedule),
              loss='binary_crossentropy',
              metrics=['accuracy'])

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
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1]
        }

    def evaluate(self, parameters, config):
        model.set_weights(parameters)

        # Evaluate on clean test data
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        y_pred_probs = model.predict(x_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        acc, recall, precision, f1 = evaluate_metrics(y_test, y_pred)

        print(f"Clean Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        return loss, len(x_test), {"accuracy": accuracy, "f1": f1,"recall":recall,"precision":precision}

# Start Federated Client
fl.client.start_numpy_client(server_address="localhost:8080", client=FlwrClient())