import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    
    # Select useful columns
    df = df[["T (degC)", "p (mbar)", "rho (g/m**3)"]]
    
    scaler = StandardScaler()
    data = scaler.fit_transform(df.values)
    
    return data

def create_sequences(data, seq_len=24):
    X, y = [], []
    
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])  # predict temperature
    
    return np.array(X), np.array(y)