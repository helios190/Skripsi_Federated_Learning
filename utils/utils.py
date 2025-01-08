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
from tf_keras.optimizers.schedules import ExponentialDecay
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping
from tf_keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

def get_model() : 
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, 30)),
        Dropout(0.5),
        Dense(65, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  
    ])

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,  
        decay_steps=1000,            
        decay_rate=0.2,               
        staircase=True                
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=5,        
        restore_best_weights=True 
    )
    return model,lr_schedule,early_stopping


def getDataset(client_id, num_clients=2, split_ratios=None,
               file_path='/Users/bintangrestubawono/Documents/skripsi_FL/Skripsi_Federated_Learning/data/creditcard.csv'):
    """
    Split the dataset into distinct portions for each client based on specified ratios.
    Each client fetches its unique portion based on `client_id`.

    Args:
    - client_id (int): ID of the client (0, 1, ..., num_clients - 1).
    - num_clients (int): Total number of clients.
    - split_ratios (list of floats): Ratios for splitting the dataset across clients.
    - file_path (str): Path to the dataset CSV file.

    Returns:
    - X_train (np.ndarray): Training features for the specific client.
    - y_train (np.ndarray): Training labels for the specific client.
    - X_test (np.ndarray): Testing features for the specific client.
    - y_test (np.ndarray): Testing labels for the specific client.
    """

    # Step 1: Load the dataset
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Step 2: Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Define split ratios if not provided
    if split_ratios is None:
        split_ratios = [1 / num_clients] * num_clients

    if len(split_ratios) != num_clients:
        raise ValueError("Number of split_ratios must match the number of clients.")

    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.")

    # Step 4: Shuffle and split the data for clients
    indices = np.arange(len(X_scaled))
    np.random.shuffle(indices)
    X_scaled = X_scaled[indices]
    y = y.iloc[indices].reset_index(drop=True)

    total_samples = len(X_scaled)
    start_idx = 0
    client_data = {}

    for i, ratio in enumerate(split_ratios):
        end_idx = start_idx + int(total_samples * ratio)
        if i == num_clients - 1:  # Ensure the last client gets the remaining samples
            end_idx = total_samples
        client_data[i] = (X_scaled[start_idx:end_idx], y[start_idx:end_idx])
        start_idx = end_idx

    # Step 5: Get the specific client's data
    if client_id not in client_data:
        raise ValueError(f"Invalid client_id: {client_id}. Must be between 0 and {num_clients - 1}.")
    X_client, y_client = client_data[client_id]

    # Step 6: Perform Random Oversampling for the client's data
    ros = RandomOverSampler(random_state=42)
    X_client_resampled, y_client_resampled = ros.fit_resample(X_client, y_client)

    # Step 7: Split the client's data into training and testing sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_client_resampled, y_client_resampled, test_size=0.2, random_state=42, stratify=y_client_resampled
    )
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    # Step 8: Return the split datasets
    return X_train, y_train.values.reshape(-1, 1), X_test, y_test.values.reshape(-1, 1)

def apply_noise_iterative(y_pred, noise_scales=[0.1, 0.5, 1, 5, 10], sensitivity=1.0, delta=1e-5):
    """
    Add Gaussian noise iteratively to predictions and calculate privacy budget.

    Args:
        y_pred (np.ndarray): Original predictions.
        noise_scales (list): List of noise standard deviations.
        sensitivity (float): Sensitivity of the predictions (default 1.0).
        delta (float): Failure probability for DP.

    Returns:
        results (dict): Noisy predictions and corresponding privacy budgets.
    """
    results = {}
    for noise_scale in noise_scales:
        # Add Gaussian noise
        noise = np.random.normal(0, noise_scale, size=y_pred.shape)
        noisy_pred = y_pred + noise
        noisy_pred = (noisy_pred > 0.5).astype(int)

        # Calculate privacy budget (epsilon)
        epsilon = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / noise_scale

        results[noise_scale] = {"noisy_pred": noisy_pred, "epsilon": epsilon}

        print(f"Noise Scale: {noise_scale}, Privacy Budget (epsilon): {epsilon:.4f}")
    return results

def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=1)
    precision = precision_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    return accuracy, recall, precision, f1

def genOutDir():
    if not os.path.exists('out'):
        os.mkdir('out')

