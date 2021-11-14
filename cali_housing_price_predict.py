from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
my_lib = __import__('1T1T_VIMARU61')

dataset = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

model = my_lib.build_linear_regression_model([X_train.shape[1]])

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('\nActual target:', y_test)
print('\nPredicted target:', y_pred)

print('\nMSE: {:2f}'.format(mean_squared_error(y_test, y_pred)))
