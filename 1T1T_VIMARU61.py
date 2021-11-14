# ==============================================================================
# Cau 1:

from tensorflow import keras
from tensorflow.keras import layers


def build_linear_regression_model(input_dim):
    '''
    Input:
        input_dim: Number of features

    Output:
        Linear Regression predict model
    '''
    model = keras.Sequential([
        layers.Dense(units=1, activation='relu', input_shape=input_dim),
    ])
    return model

# ==============================================================================
# Cau 2:

from sklearn.neighbors import KNeighborsClassifier


def train_pred(X_train, y_train, X_test, k):
    '''
    Input: 
        X_train: Training data
        y_train: Training target
        X_test: Test data
        k: Number of neighbors to use
        
    Output:
        y_test: Predicted target of X_test
    '''
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(X_train, y_train)
    return model.predict(X_test)
