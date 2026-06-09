"""
Advanced RNN (LSTM) Deep Learning Example
Task: Time Series Forecasting with TensorFlow/Keras

Features:
- Data normalization
- Sequence generation
- Stacked LSTM layers
- Dropout regularization
- EarlyStopping callback
- ModelCheckpoint
- Prediction and evaluation

Install:
pip install tensorflow pandas numpy scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Generate Sample Data
# ---------------------------
np.random.seed(42)
time = np.arange(0, 1000)
data = np.sin(0.02 * time) + np.random.normal(0, 0.1, len(time))

df = pd.DataFrame(data, columns=["value"])

# ---------------------------
# Normalize Data
# ---------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# ---------------------------
# Create Sequences
# ---------------------------
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 30
X, y = create_sequences(scaled_data, SEQ_LEN)

# Train-Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------------------
# Build Advanced LSTM Model
# ---------------------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ---------------------------
# Callbacks
# ---------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ---------------------------
# Train Model
# ---------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------
# Predictions
# ---------------------------
predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test)

# ---------------------------
# Evaluation
# ---------------------------
rmse = np.sqrt(mean_squared_error(actual, predictions))
print(f"RMSE: {rmse:.4f}")

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(12,6))
plt.plot(actual, label="Actual")
plt.plot(predictions, label="Predicted")
plt.title("LSTM Time Series Forecasting")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

# Save model
model.save("advanced_rnn_lstm_model.h5")

print("Model saved successfully!")
