import pandas as pd
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def prepare_data():
    # 1. load data
    X, y = datasets.load_diabetes(return_X_y=True)
    X = X[:, 1:3]

    # 2. Split Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=5
    )
    return X_train, X_test, y_train, y_test


def train(X, y):
    # 3. Fit model
    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    return reg


def predict(reg, X):
    # 4. Predict
    pred = reg.predict(X)

    return pred


def main():
    # 1. prepare data
    X_train, X_test, y_train, y_test = prepare_data()

    # 2. train
    reg = train(X_train, y_train)

    # 3. predict
    y_pred = predict(reg, X_test)
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(reg.coef_)


if __name__ == "__main__":
    main()
