import torch
import numpy as np
from model import LSTMModel
from utils import mape

def train_model(X_train, y_train, X_test, y_test):

    model = LSTMModel(input_size=X_train.shape[2])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()

        output = model(X_train).squeeze()
        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32)
    preds = model(X_test).detach().numpy().flatten()

    print("MAPE:", mape(y_test, preds))

    return preds