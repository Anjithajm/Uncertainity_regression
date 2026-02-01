import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# Generate synthetic dataset
# -----------------------------
np.random.seed(42)

N = 2000
X = np.random.uniform(-5, 5, size=(N, 1))
y = np.sin(X) + 0.3 * np.random.randn(N, 1)

# Train / test split
split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# Model with dropout
# -----------------------------
model = models.Sequential([
    layers.Dense(64, activation="relu", input_shape=(1,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(X_train, y_train, epochs=50, verbose=0)

# -----------------------------
# Monte Carlo Dropout inference
# -----------------------------
def mc_dropout_predict(model, X, n_samples=100):
    preds = [model(X, training=True) for _ in range(n_samples)]
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)

mean_pred, uncertainty = mc_dropout_predict(model, X_test)

print("Mean prediction shape:", mean_pred.shape)
print("Uncertainty shape:", uncertainty.shape)
print("Example prediction:", mean_pred[0][0], "+/-", uncertainty[0][0])
