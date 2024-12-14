import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tf_keras.models import Sequential,Model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tf_keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Add, BatchNormalization, Activation,LSTM, Dense, Dropout
from tf_keras.optimizers import Adam
from tf_keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

def plot_metrics(metrics, metric_names, title="Metrics Comparison"):
    num_clients = len(metrics)
    x = np.arange(len(metric_names))

    # Plot each client's metrics
    for i, client_metrics in enumerate(metrics):
        plt.bar(x + i * 0.2, client_metrics, width=0.2, label=f'Client {i+1}')

    plt.xticks(x + 0.2 * (num_clients - 1) / 2, metric_names)
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.show()

# Load and compile Keras model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, 30)),
    Dropout(0.5),
    Dense(65, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

def getDataset(client_id, num_clients=2):
    """
    Split the dataset into distinct portions for each client.
    Each client fetches its unique portion based on `client_id`.

    Args:
    - client_id (int): ID of the client (0, 1, ..., num_clients - 1).
    - num_clients (int): Total number of clients.

    Returns:
    - X_train, y_train, X_test, y_test: Split for the specific client.
    """
    df = pd.read_csv('/Users/bintangrestubawono/Documents/fl7/creditcard.csv')
    X = df.drop(columns=['Class'])
    y = df['Class']
    X = np.expand_dims(X.values, axis=1)  # Reshape for LSTM

    # Split the dataset into `num_clients` distinct parts
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Calculate split sizes
    client_data_size = len(X) // num_clients
    start_idx = client_id * client_data_size
    end_idx = (client_id + 1) * client_data_size if client_id != num_clients - 1 else len(X)

    client_indices = indices[start_idx:end_idx]

    X_client = X[client_indices]
    y_client = y.iloc[client_indices]

    # Further split into train and test for the client
    X_train, X_test, y_train, y_test = train_test_split(X_client, y_client, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

def apply_noise(y_pred, noise_scale=0.1):
    noise = np.random.normal(0, noise_scale, size=y_pred.shape)
    noisy_pred = y_pred + noise
    noisy_pred = (noisy_pred > 0.5).astype(int)
    return noisy_pred

def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=1)
    precision = precision_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    return accuracy, recall, precision, f1

def genOutDir():
    if not os.path.exists('out'):
        os.mkdir('out')

