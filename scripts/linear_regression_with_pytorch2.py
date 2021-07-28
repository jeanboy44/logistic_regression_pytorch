import torch

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

EPOCHS = 500000
LEARNING_RATE = 0.005


def prepare_data():
    # 1. load data
    X, y = datasets.load_diabetes(return_X_y=True)
    X = X[:, 1:3]
    # X = np.concatenate([np.ones(X.shape[0]).reshape(-1,1), X], axis=1)

    # 2. Split Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=5
    )

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float().view(-1, 1)
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float().view(-1, 1)

    return X_train, X_test, y_train, y_test


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # bias term is automatically added.

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def train(model, X, y, criterion, optimizer):
    model.train()
    # Forward pass
    y_pred = model(X)
    # Compute Loss
    cost = criterion(y_pred, y)
    # Backward pass
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    return cost


def main():
    # 1. prepare data
    X_train, X_test, y_train, y_test = prepare_data()

    # 2. Fit model
    # initialize
    model = LinearRegression()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # fit
    for epoch in range(EPOCHS):
        cost = train(model, X_train, y_train, criterion, optimizer)
        # print
        if epoch % 100000 == 0:
            print(f"Epoch {epoch}/{EPOCHS} Cost: {cost.item()}")

    # 5. Predict
    y_pred = model(X_test)
    print(f"MSE: {mean_squared_error(y_test, y_pred.detach().numpy())}")
    theta1_ = model.linear.weight.detach().numpy().flatten()  # get weigths from model.
    theta0 = (
        model.linear.bias.detach().numpy().flatten()
    )  # get bias(intercept) weight from model.
    theta = np.concatenate([theta0, theta1_])  # concat weights
    print(theta)


if __name__ == "__main__":
    main()
