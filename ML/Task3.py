import pandas as pd
from pandas import read_csv
from matplotlib.pyplot import plot, title, xlabel, ylabel, legend, show
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings

dataset = pd.read_csv('C:\\Users\\Tom\\Desktop\\ML\\Task3_dataset_HIV_RVG.csv')

df = pd.read_csv('C:\\Users\\Tom\\Desktop\\ML\\Task3_dataset_HIV_RVG.csv')
df_val = df.values
# assigns all the columns except the last column to the variable X
X = df_val[:,0:7]
# assigns only the last column as variable Y
Y = df_val[:,8]

#LabelEncoder encodes target labels with value between 0 and n_classes-1. (scikit-learn.org, n.d.)
#it can be used to normalize labels and transform non-numerical labels to numerical labels (scikit-learn.org, n.d.)
#The status column is not numerical and since machine learning models are based on mathematical equations,
#categorical variables need to be changed to numbers.
#OneHotEncoder encodes categorical features as a one-hot numeric array. (scikit-learn.org, n.d.)
#So, the data in the status column is transformed to numerical values, then put into an array.
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

print(df.describe())
print('Size of data:', dataset.shape)
print('Count of data with Participant Condition \"Patient\":',
      dataset[dataset['Participant Condition'] == 'Patient'].index.size)
print('Count of data with Participant Condition \"Control\":',
      dataset[dataset['Participant Condition'] == 'Control'].index.size)
print('Count of missing values:', dataset.isna().any().sum())

# replace the data of the column "Participant Condition" with 1 ("Patient") or 0 ("Control")
dataset.loc[dataset['Participant Condition'] == 'Patient', 'Participant Condition'] = 1
dataset.loc[dataset['Participant Condition'] == 'Control', 'Participant Condition'] = 0

dataset = shuffle(dataset)  # shuffle the data

# use the data of all columns except the column "Participant Condition" as the input values
x = dataset.drop(['Participant Condition'], axis=1).astype('int')

# use the data of the column "Participant Condition" as the output values
y = dataset['Participant Condition'].astype('int')

x_Normalised = StandardScaler().fit_transform(x).astype('int')  # Normalise the data

# a box plot which will include "Participant Condition" in the x-axis and "Alpha" in the y-axis
dataset.boxplot(column=['Alpha'], by='Participant Condition')
title(None)
xlabel('Control                                         Patient')
ylabel('Alpha')
show()

# a density plot for "Beta" of each Participant Condition
dataset[dataset['Participant Condition'] == 1]['Beta'].plot(kind='kde', label='Patient')
dataset[dataset['Participant Condition'] == 0]['Beta'].plot(kind='kde', label='Control')
xlabel('Beta')
legend()
show()

# split the data into training (90% of the data) and test sets (10% of the data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Normalise the split data which will be used to apply an ANN
scaler = StandardScaler().fit(x_train)
x_train_Normalised = scaler.transform(x_train).astype('int')
x_test_Normalised = scaler.transform(x_test).astype('int')


def ApplyAnn(x_train, x_test, y_train, y_test, epoch=200) -> float:
    """
    Design, train, and evaluate a specified ANN with 500 neurons in the hidden layer.

    x_train: training input values
    x_test: test input values
    y_train: training output values
    y_test: test output values
    epoch: an integer representing the max iteration steps (default = 200)
    return: the accuracy of the ANN
    """

    ann = MLPClassifier(activation='logistic', hidden_layer_sizes=(500, 500), max_iter=epoch, random_state=1)
    ann.fit(x_train, y_train)

    return ann.score(x_test, y_test)
print('Accuracy:', ApplyAnn(x_train_Normalised, x_test_Normalised, y_train, y_test))

filterwarnings('ignore')  # the function is mainly used to ignore convergence warnings
epochList = [10 * i for i in range(1, 11)]
accuracyList = []

# loop to use various epochs to design, train, and evaluate an ANN
for epoch in epochList:
    accuracyList.append(ApplyAnn(x_train_Normalised, x_test_Normalised, y_train, y_test, epoch))

print('List of Epochs:', epochList)
print('List of Accuracy:', accuracyList)

plot(epochList, accuracyList)
xlabel('Epoch')
ylabel('Accuracy')
show()

def ApplyRandomForest(x_train, x_test, y_train, y_test, treeNum=10, minSampleNum=1) -> float:
    """
    Design, train, and evaluate a specified random forest classifier.

    x_train: training input values
    x_test: test input values
    y_train: training output values
    y_test: test output values
    treeNum: the number of trees (default = 10)
    minSampleNum: the minimum number of samples required to be at a leaf node (default = 1)
    returns: the accuracy of a random forest classifier
    """

    randomForest = RandomForestClassifier(criterion='entropy', min_samples_leaf=minSampleNum, n_estimators=treeNum,
                                          random_state=1)
    randomForest.fit(x_train, y_train)
    return randomForest.score(x_test, y_test)


print('Accuracy:', ApplyRandomForest(x_train, x_test, y_train, y_test, 1000, 5))
print('Accuracy:', ApplyRandomForest(x_train, x_test, y_train, y_test, 1000, 10))

treeNumList = [10, 50, 100, 500, 1000]
accuracyList_5 = []
accuracyList_10 = []

# loop to use various number of trees to design, train, and evaluate a random forest classifier
for treeNum in treeNumList:
    accuracyList_5.append(ApplyRandomForest(x_train, x_test, y_train, y_test, treeNum, 5))
    accuracyList_10.append(ApplyRandomForest(x_train, x_test, y_train, y_test, treeNum, 10))

print('List of numbers of trees:', treeNumList)
print('List of accuracy (at least 5 samples at a leaf node):', accuracyList_5)
print('List of accuracy (at least 10 samples at a leaf node):', accuracyList_10)

plot(treeNumList, accuracyList_5, label='at least 5 samples at a leaf node')
plot(treeNumList, accuracyList_10, label='at least 10 samples at a leaf node')
xlabel('Number of Trees')
ylabel('Accuracy')
legend()
show()

tenFoldCv = KFold(n_splits=10, random_state=None, shuffle=False)  # apply a 10-fold CV process

# loop to show the mean accuracy results for each set of parameters of an ANN
for neuronNum in [50, 500, 1000]:
    cvScore_Ann = cross_val_score(
        MLPClassifier(activation='logistic', hidden_layer_sizes=(neuronNum, neuronNum), random_state=1), x_Normalised,
        y, cv=tenFoldCv)
    print('Mean accuracy of the ANN with {0} neurons in each hidden layer: {1}'.format(neuronNum, cvScore_Ann.mean()))

print()

# loop to show the mean accuracy results for each set of parameters of a random forest classifier
for treeNum in [50, 500, 10000]:
    cvScore_RandomForest = cross_val_score(
        RandomForestClassifier(criterion='entropy', n_estimators=treeNum, random_state=1), x, y, cv=tenFoldCv)
    print('Mean accuracy of the random forest classifier with {0} trees: {1}'.format(treeNum,
                                                                                     cvScore_RandomForest.mean()))
