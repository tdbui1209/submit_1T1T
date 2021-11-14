from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
my_lib = __import__('1T1T_VIMARU61')

dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)

y_pred = my_lib.knn_train_pred(X_train, y_train, X_test, 5)

y_actual = [dataset.target_names[i] for i in y_test]
print('\nActual target:', y_actual)

y_pred = [dataset.target_names[i] for i in y_pred]
print('\nPredicted target:', y_pred)

print('\nAccuracy score: {:.2f}'.format(accuracy_score(y_actual, y_pred)))
