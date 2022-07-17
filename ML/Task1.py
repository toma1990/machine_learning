import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:\\Users\\Tom\\Desktop\\ML\\Task1_dataset_pol_regression.csv')
print(dataset)


# function that does the feature expansion up to a certain degree for a data set.
def getPolDataMatrix(features, degree):  # original features of the data and a certain degree of a polynomial
    """
    features: original features of the data
    degree: degree of polynomial
    return: a data matrix for the polynomial
    """
    dataMatrix = []

    for i in range(degree + 1):  # range: [0, degree + 1)
        dataMatrix.append(features ** i)

    return np.mat(dataMatrix).transpose()  # data matrix for a polynomial


# Regress a polynomial of a certain degree
def pol_regression(features_train, y_train, degree):
    """
    features_train: Training input values given as a 1-D NumPy array
    y_train: Training output values given as a 1-D NumPy array
    return: a NumPy array of the parameters of a polynomial
    """

    x = getPolDataMatrix(features_train, degree)  # do the polynomial feature expansion
    xTx = x.transpose().dot(x)
    y = y_train.reshape((len(y_train), 1))
    parameters = np.linalg.solve(xTx, x.transpose().dot(y))  # the least-squares solution
    return parameters


x_train = np.array(dataset['x'])
y_train = np.array(dataset['y'])

for degree in [0, 1, 2, 3, 6, 10]:
    parameters = pol_regression(x_train, y_train, degree)  # regress a polynomial of a certain degree
    x_range = [(-5 + count * 0.02) for count in range(int((5 - (-5)) // 0.02) + 1)]  # the range of x: [-5, 5]
    dataMatrix = getPolDataMatrix(np.array(x_range), degree)  # do the polynomial feature expansion
    y_pred = dataMatrix.dot(parameters)

    if degree == 10:
        plt.ylim(min(y_train) - 20, max(y_train) + 20)

    # plot the resulting polynomial
    plt.title('Degree = {}'.format(degree))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_train, y_train, label='Training Points')
    plt.plot(x_range, y_pred, 'Red', label='Polynomial Regression')
    plt.legend()
    plt.show()


# Evaluate the performance of a polynomial.
def eval_pol_regression(parameters, x, y, degree):
    """
    parameters: the parameters of a polynomial
    x: input values given as a 1-D NumPy array
    y: output values given as a 1-D NumPy array
    degree: a certain degree of a polynomial
    return RMSE: the root mean squared error (RMSE) of a polynomial
    """

    dataMatrix = getPolDataMatrix(x, degree)  # do the polynomial feature expansion
    y_pred = np.array(dataMatrix.dot(parameters))
    y_pred = y_pred.reshape((len(y_pred)))  # reshape the 2-D NumPy array y_pred to a 1-D NumPy array
    RMSE = np.sqrt(np.mean(np.power(y - y_pred, 2)))

    return RMSE


TrainingSet = dataset.sample(frac=0.7)
TestSet = dataset[~dataset.index.isin(TrainingSet.index)]
x_train = np.array(TrainingSet['x'])
y_train = np.array(TrainingSet['y'])
x_Test = np.array(TestSet['x'])
y_Test = np.array(TestSet['y'])
degreeList = [0, 1, 2, 3, 6, 10]
RMSEList_train = []
RMSEList_Test = []

for degree in degreeList:
    parameters = pol_regression(x_train, y_train, degree)  # regress a polynomial of a certain degree
    x_range = [(-5 + count * 0.02) for count in range(int((5 - (-5)) // 0.02) + 1)]  # the range of x: [-5, 5]
    dataMatrix = getPolDataMatrix(np.array(x_range), degree)  # do the polynomial feature expansion
    y_pred = dataMatrix.dot(parameters)

    RMSE_train = eval_pol_regression(parameters, x_train, y_train, degree)  # evaluate the Training set RMSE
    RMSE_Test = eval_pol_regression(parameters, x_Test, y_Test, degree)  # evaluate the Test set RMSE
    RMSEList_train.append(RMSE_train)
    RMSEList_Test.append(RMSE_Test)

    if degree == 10:
        plt.ylim(min(dataset['y']) - 20, max(dataset['y']) + 20)

    # plot the resulting polynomial
    plt.title('Degree = {}\nTraining Set RMSE = {}\nTest Set RMSE = {}'.format(degree, RMSE_train, RMSE_Test))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_train, y_train, label='Training Points')
    plt.scatter(x_Test, y_Test, c='Orange', label='Test Points')
    plt.plot(x_range, y_pred, 'Green', label='Polynomial Regression')
    plt.legend()
    plt.show()

    # plot both RMSE values
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.plot(degreeList, RMSEList_train, label='Training Set RMSE')
plt.plot(degreeList, RMSEList_Test, label='Test Set RMSE'
                                          '')
plt.legend()
plt.show()
