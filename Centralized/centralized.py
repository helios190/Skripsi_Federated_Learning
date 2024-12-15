from tf_keras.models import Sequential
import pandas as pd
import numpy as np
from tf_keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tf_keras.optimizers.schedules import ExponentialDecay
from tf_keras.callbacks import EarlyStopping
from tf_keras.optimizers import Adam
import sys
from utils.utils import evaluate_metrics, apply_noise_iterative

# Load the dataset
df = pd.read_csv('./data/creditcard.csv')
X = df.drop(columns=['Class'])
y = df['Class']
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Apply StandardScaler to standardize the data
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Expand dimensions for LSTM compatibility
X_resampled_scaled = np.expand_dims(X_resampled_scaled, axis=1)

# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled_scaled, 
    y_resampled, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_resampled  # Maintain class proportions in train-test split
)

# Ensure y_train and y_test are reshaped correctly
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# Build the model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, 30)),
    Dropout(0.5),
    Dense(65, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
# Define the learning rate scheduler
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,  
    decay_steps=1000,            
    decay_rate=0.2,               
    staircase=True                
)

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=5,          # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore the best model weights
)

# Use the scheduler in the optimizer
model.compile(
    optimizer=Adam(learning_rate=lr_schedule),  # Add the scheduler here
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model with validation
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    verbose=1,
    validation_data=(X_test, y_test)  # Validation during training
)

# Export training history to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
# Original predictions
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# Apply noise iteratively
noisy_predictions = apply_noise_iterative(y_pred, noise_scales=[0.1, 0.5, 1, 5, 10])

# Initialize results list for metrics
metrics_list = []

# Evaluate metrics for each noise scale
for noise_scale, noisy_pred in noisy_predictions.items():
    noisy_accuracy, recall, precision, f1 = evaluate_metrics(y_test, noisy_pred)
    metrics_list.append({
        "Noise Scale": noise_scale,
        "Noisy Accuracy": noisy_accuracy,
        "Recall": recall,
        "Precision": precision,
        "F1": f1
    })

# Convert metrics to a DataFrame and export to CSV
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv('noisy_evaluation_metrics.csv', index=False)

print("Noisy evaluation metrics saved to 'noisy_evaluation_metrics.csv'")