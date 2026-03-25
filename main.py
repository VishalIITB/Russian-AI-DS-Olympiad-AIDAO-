from src.data_loader import load_data, create_sequences
from src.quantum_features import quantum_feature_map
from src.train import train_model
import numpy as np

# Load data
data = load_data("jena_climate_2009_2016.csv")

# Create sequences
X, y = create_sequences(data)

# Apply quantum features
X = quantum_feature_map(X)

# Train-test split
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
preds = train_model(X_train, y_train, X_test, y_test)