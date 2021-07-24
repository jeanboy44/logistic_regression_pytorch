# 1주차. 로지스틱 회귀모형을 이용한 딥러닝 구성요소 설명(with code)

# Simple Example

Jupyter Notebook

# Prediction

## **Mathematical formula**

- 기호 설명
    - w: 가중치 벡터(Dx1 차원)
    - x: Input 벡터(1xD 차원)
    - b: 절편 항

![https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532479726851_image.png](https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532479726851_image.png)

![https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532480054025_image.png](https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532480054025_image.png)

- 선형 회귀

$$y=w^Tx+b$$

- 로지스틱 회귀(선형회귀식에 시그모이드 함수를 씌운 형태)

    $$y={\sigma}(w^Tx+b), {\sigma}(a)=\frac{1}{1+exp(-a)}$$

$$y=\frac{1}{1+exp(-(w^Tx+b))}$$

## **Code**

- 선형 회귀
    - Loop로 표현 - 느림

```python
prediction = 0
for i in range(D):
    prediction += w[i]*x[i]
    prediction = prediction+b
```

- 벡터화(numpy) - 빠름

```
w.dot(x)+b # w 내적 x
```

- 로지스틱 회귀
    - Loop로 표현 - 느림

```
prediction = 0
for i in range(D):
    prediction += w[i]*x[i]
    prediction = sigmoid(prediction+b)
```

- 벡터화(numpy) - 빠름

```
sigmoid(w.dot(x)+b) # w 내적 x
```

- 로지스틱 회귀(여러 $$x$$를 예측)
    - 각 x 에 대해 Loop
        - x : 1 x D 차원
        - X : N x D 차원

```
predictions = []
for i in range(N):
    p = sigmoid(w.dot(X[i]) +b))
    predictions.append(p)
```

- 벡터화

```
predictions = sigmoid(X.dot(w) + b) # X와 w의 행렬곱 ( X(NxD) x w(Dx1) = P(Nx1))
```

## **Logistic Regression in neural network**

![https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532506813102_image.png](https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532506813102_image.png)

![https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532485137085_logistic_regression_activation.jpg](https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532485137085_logistic_regression_activation.jpg)

[https://camo.githubusercontent.com/4101184d2eb8556f1c2ecde71cc8af426863060c/68747470733a2f2f61373061643264313639393638323065363238352d33633331353436323937363334336439303364356233613033623639303732642e73736c2e6366322e7261636b63646e2e636f6d2f6664346261396537653638623736666334316338333132383536633764306164](https://camo.githubusercontent.com/4101184d2eb8556f1c2ecde71cc8af426863060c/68747470733a2f2f61373061643264313639393638323065363238352d33633331353436323937363334336439303364356233613033623639303732642e73736c2e6366322e7261636b63646e2e636f6d2f6664346261396537653638623736666334316338333132383536633764306164)

- Activation function
    - Neural network가 끝나는 시점에 아웃풋에 붙이는 것(노드). Transfer Function이라고도 불리며, 두 개의 Neural Networks의 사이에도 들어갈 수 있음
    - Neural network의 결과가 “YES” 또는 “NO” 인지를 결정하기 위해 사용하며, 아웃풋 값들을 0 ~ 1 또는 -1 ~ 1 범위의 값으로 매핑한다.
- Activation function의 종류
    - Step function
- 값이 0 이상이면 1, 아니면 0을 반환하는 함수

![https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532484056065_step_funtion.png](https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532484056065_step_funtion.png)

- Linear function

$$f(x)=ax$$

- Sigmoid function

$$f(x) = \frac{1}{1+exp(-x)}$$

![https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532484307266_sigmoid.png](https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532484307266_sigmoid.png)

- Tanh function

$$f(x)=tanh(x)=\frac{2}{1+exp(-2x)}-1$$

$$f(x)=tanh(x)=2sigmoid(2x)-1$$

- ReLu

$$f(x)=max(0,x)$$

![https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532484445005_ReLu.jpg](https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532484445005_ReLu.jpg)

# Training

- 기계학습의 “학습”은 단순히 모델의 가중치(w)를 찾아내는 것
    - 비유하자면, 새로운 기억이 생성될 때마다, 뇌에 있는 각 시냅스 간의 연결의 세기가 변한다!
- 이러한 관점에서, 기계학습 문제는 단순히 주어진 데이터(X, y)를 가장 잘 설명하는 가중치를 찾아내는 최대우도(Maximum likelihood) 문제를 해결하는 것이다.
- Ex) 동전던지기 데이터를 바탕으로 동전의 앞면이 나올 확률을 학습

$$likelihood = p(heads)p(heads)...p(tails)p(tails)$$

$$p(heads) = \frac{\#\,heads}{\#\,total\,coin\,tosses}$$

- Ex) 동전던지기 데이터를 바탕으로 동전의 앞면이 나올 확률을 학습(같은 식)

$$likelihood = w^{n}(1-w)^{N-n}$$

$$w = \frac{n}{N-n}$$

## **Cost function / Loss function**

- 결국 cost function/loss function이란, 주어진 데이터에 대해서 선택된 가중치가 적절한지 아닌지를 평가할 수 있는 척도를 계산하는 식이라고 해석할 수 있으며, 그 척도(likelihood = 가능도)를 최대화하는 가중치를 찾는 과정이 학습임
- Cost function의 대표적인 예: **Binary cross-entropy**
    - Negative log-likelihood이며, likelihood에 로그를 취하고 마이너스를 붙임

$$J = -\{\sum_{n=1}^Nt_nlogy_n+(1-t_n)log(1-y_n)\}$$

## **Optimization Method**

- 위에 정의한 cost function을 최대화 하기위해서는 어떠한 과정이 필요한가?
- 일반적으로는 수리적인 방법(Ex. 동전던지기의 확률을 구한 방법, 미분 등)으로 해를 구할 수 없음
- Optimization Method의 대표적인 예: **Gradient descent**
    - Mathematical Formula

$$w \leftarrow w - \eta{\nabla}_wJ$$

$${\nabla}_nJ=X^T(Y-T)=\sum_{n=1}^Nx_n(y_n-t_n)$$

![https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532506009456_image.png](https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532506009456_image.png)

- Code

```
gradient_w = X.T.dot(Y-T)
```

- 실제 학습 시에는, 업데이트 횟수를 정하거나, loss(cost)의 값의 제한을 둔 후 loop를 돌림

```
for epoch in range(num_epochs):
    w = w - learning_rate*X.T.dot(Y-T)
```

## **Regulaziation**

- 모델의 가중치가 너무 크게되지 않도록 조정하는 것이며, 이를 베이지안 확률 관점으로 해석하자면, MLE문제를 MAP문제로 바꾸어서 해결하는 것임

![https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532484307266_sigmoid.png](https://d2mxuefqeaa7sj.cloudfront.net/s_66D6ABF14247BB2B47EF457B27E45A6F278026C82A1A87288E73E8AA188FA574_1532484307266_sigmoid.png)

- Ex)

$$J = -\{\sum_{n=1}^Nt_nlogy_n+(1-t_n)log(1-y_n)\}$$

- J를 최소화하는 w 를 구하는 과정이 학습이며,
    - t_n(실제 타겟)에 1일 때,
        - y_n가 1이어야 J가 최소화되고,

        $$y=\frac{1}{1+exp(-(w^Tx+b))}$$

        - 이므로, w 가 무한대가 돼야함.

- L1 regularization(encourages sparsity)

$$J_{L1} = J + {\lambda}_1|w|$$

- 일부 가중치가 0의 값을 가지게 됨

- L2 regularization(encourages small weights)

$$J_{L2} = J + {\lambda}_2|w|^2$$

- 가중치가 전체적으로 작은 값을 가지게 됨

## Simple Example Again!

**Keras Code**

```
from keras.regularizers import L1L2
reg = L1L2(l1=0.0001, l2=0.0001)

## Build
model = Sequential()
model.add(Dense(activation='sigmoid', input_dim=X_train.shape[1], units=Y_train.shape[1], W_regularizer=reg, ))

## Complie
model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

## Fit
model.fit(X_train, Y_train, nb_epoch=10, validation_split=0.7)

## Evaluate
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
```

**Jupyter Notebook**

**참고문헌**

[Data Science: Deep Learning in Python] / Lazy Programmer /Udemy

[http://bjlkeng.github.io/posts/probabilistic-interpretation-of-regularization/](http://bjlkeng.github.io/posts/probabilistic-interpretation-of-regularization/)

[https://math.stackexchange.com/questions/477207/derivative-of-cost-function-for-logistic-regression](https://math.stackexchange.com/questions/477207/derivative-of-cost-function-for-logistic-regression)

[https://github.com/dusty-nv/jetson-inference/blob/master/docs/deep-learning.md](https://github.com/dusty-nv/jetson-inference/blob/master/docs/deep-learning.md)

[https://stackoverflow.com/questions/26058022/neural-network-activation-function-vs-transfer-function](https://stackoverflow.com/questions/26058022/neural-network-activation-function-vs-transfer-function)

[https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

[https://gist.github.com/fchollet/b7507f373a3446097f26840330c1c378](https://www.notion.so/b7507f373a3446097f26840330c1c378)

[https://gist.github.com/fchollet/b7507f373a3446097f26840330c1c378](https://www.notion.so/b7507f373a3446097f26840330c1c378)

[https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970](https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970)

**Gradient 계산**

[https://math.stackexchange.com/questions/477207/derivative-of-cost-function-for-logistic-regression](https://math.stackexchange.com/questions/477207/derivative-of-cost-function-for-logistic-regression)