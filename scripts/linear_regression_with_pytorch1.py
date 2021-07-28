import torch
import torch.optim as optim

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

EPOCHS = 500000
LEARNING_RATE = 0.005


def predict(X, theta):
    hypothesis = torch.matmul(X, theta)

    return hypothesis


def prepare_data():
    # 1. load data
    X, y = datasets.load_diabetes(return_X_y=True)
    X = X[:, 1:3]
    X = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1), X], axis=1)  # Add bias term

    # 2. Split Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=5
    )

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    return X_train, X_test, y_train, y_test


def train(X, y, theta, optimizer):
    # H(x) 계산
    hypothesis = torch.matmul(X, theta)

    # cost 계산
    cost = torch.mean((hypothesis - y) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    return cost


def main():
    # 1. prepare data
    X_train, X_test, y_train, y_test = prepare_data()
    N = X_train.shape[0]
    D = X_train.shape[1]

    # 2. train model
    # initailize
    theta = torch.zeros(D, requires_grad=True).float()
    optimizer = optim.SGD([theta], lr=LEARNING_RATE)  # Stochastic Gradient Descenct
    # fit
    for epoch in range(EPOCHS):
        cost = train(X_train, y_train, theta, optimizer)
        # 100번마다 로그 출력
        if epoch % 100000 == 0:
            print(f"Epoch {epoch}/{EPOCHS} Cost: {cost.item()}")

    # 3. predict
    y_pred = predict(X_test, theta)
    print(f"MSE: {mean_squared_error(y_test, y_pred.detach().numpy())}")
    print(theta.detach().numpy())


if __name__ == "__main__":
    main()
