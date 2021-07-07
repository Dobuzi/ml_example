from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from matplotlib import pyplot as plt
import numpy as np
 
def makeDataset(dataset, slice_ratio):
    slicer = int(dataset[0].shape[0] * slice_ratio)

    data_X, data_y = dataset
    data_X = data_X[:, np.newaxis, 2]

    data_X_train, data_X_test = data_X[:slicer], data_X[slicer:]
    data_y_train, data_y_test = data_y[:slicer], data_y[slicer:]
    
    return (data_X_train, data_y_train), (data_X_test, data_y_test)

if __name__ == '__main__':
    dataset = datasets.load_diabetes(return_X_y=True)
    (data_train_X, data_train_y), (data_test_X, data_test_y) = makeDataset(dataset, 0.8)

    regr = linear_model.LinearRegression()
    regr.fit(data_train_X, data_train_y)

    prediction = regr.predict(data_test_X)
    error = mean_squared_error(data_test_y, prediction)

    print(f'Coefficients: {regr.coef_}')
    print(f'Mean squared error: {error}')

    plt.scatter(data_test_X, data_test_y, color='black')
    plt.plot(data_test_X, prediction, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
