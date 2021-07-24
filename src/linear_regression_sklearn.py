import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def prepare_data():
    # 1. load data
    cols = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
    ]
    # https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/housing/housing.data
    df = pd.read_csv(
        "data/raw/housing.data.txt",
        delimiter=r"\s+",
        names=cols,
    )
    df = df.dropna()
    X = df.drop("MEDV", axis=1)
    y = df[["MEDV"]]

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
    X_train, X_test, y_train, y_test = prepare_data()
    reg = train(X_train, y_train)
    y_pred = predict(reg, X_test)
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")


if __name__ == "__main__":
    main()
