import numpy as np

def quantum_feature_map(X):
    # Quantum-inspired encoding
    return np.concatenate([np.sin(X), np.cos(X)], axis=2)