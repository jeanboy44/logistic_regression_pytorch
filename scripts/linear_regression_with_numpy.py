import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

EPOCHS = 500000
LEARNING_RATE = 0.005


# Gradient Descent 함수 생성
def gradientDescent(X, y, theta, alpha, N, numIterations, verbose=1):
    X_tmp = X.copy()
    X_tmp = np.concatenate(
        [np.ones(X_tmp.shape[0]).reshape(-1, 1), X_tmp], axis=1
    )  # Add bias term
    #     tmp = (tmp - tmp.mean()) / (tmp.max() - tmp.min()) # standardization for computing convinience
    # bias 추가
    for i in range(0, numIterations):
        # Predict
        hypothesis = X_tmp.dot(theta)
        loss = hypothesis.reshape(-1, 1) - y.reshape(-1, 1)
        # avg cost per example (the 2 in 2*n doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / N
        if verbose == 1:
            if i % 100000 == 0:
                print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = X_tmp.T.dot(loss) / N
        # update
        theta = theta - alpha * gradient
    return theta


def prepare_data():
    # 1. load data
    X, y = datasets.load_diabetes(return_X_y=True)
    X = X[:, 1:3]

    # 2. Split Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=5
    )
    return X_train, X_test, y_train, y_test


def train(X, y, epochs, learning_rate):
    # - Initialize weight
    D = X.shape[1] + 1
    theta = np.random.normal(0, 1, D).reshape(-1, 1)  # Initalize theta

    # 4. Fit model
    theta = gradientDescent(
        X=X,
        y=y,
        theta=theta,
        alpha=learning_rate,
        N=X.shape[0],
        numIterations=epochs,
        verbose=1,
    )

    return theta


def predict(X, theta):
    X = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1), X], axis=1)  # Add bias term
    y_hat = X.dot(theta)

    return y_hat.flatten()


def main():
    # 1. pepare data
    X_train, X_test, y_train, y_test = prepare_data()

    # 2. train model
    theta = train(X_train, y_train, EPOCHS, LEARNING_RATE)

    # 3. predict
    y_pred = predict(X_test, theta)
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(theta)


if __name__ == "__main__":
    main()
