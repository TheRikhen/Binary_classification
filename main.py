import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

data_train = pd.read_csv('winequality-red.csv', delimiter=",")
print('Number of rows and columns in initial dataset:', data_train.shape)
train, validation = train_test_split(data_train, test_size=0.2)
arr_name = []
arr_train = []
arr_val = []
# используем предварительно отобранные признаки
cols_x = ['volatile acidity', 'alcohol', 'citric acid', 'sulphates']
# целевой признак
col_y = 'quality'


def show_correlation_heatmap():
    sns.set(style="ticks")
    sns.heatmap(data_train.corr())
    plt.show()


def classifier_testing(classifier, classifier_name):
    classifier.fit(train[cols_x], train[col_y])
    y_train = classifier.predict(train[cols_x])
    y_train_acc = accuracy_score(train[col_y], y_train)
    y_val = classifier.predict(validation[cols_x])
    y_val_acc = accuracy_score(validation[col_y], y_val)
    arr_name.append(classifier_name)
    arr_train.append(y_train_acc)
    arr_val.append(y_val_acc)
    print('Accuracy per {} algorithm on train data = {}, on validation data = {}' \
          .format(classifier_name,
                  round(y_train_acc, 3),
                  round(y_val_acc, 3)))
    return classifier


def classifier_testing_comparison():
    x = range(len(arr_train))
    plt.plot(x, arr_train)
    plt.plot(x, arr_val)
    plt.xticks(x, arr_name)
    plt.ylabel('algorithm accuracy')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


def predict_model():
    predicted_good_wine = []
    is_good_wine = []
    srwg, srwb, sfwg, sfwb = 0, 0, 0, 0
    clf = DecisionTreeClassifier()
    clf.fit(train[cols_x], train[col_y])
    for i in validation[col_y]:
        is_good_wine.append(i)
    for i in clf.predict(validation[cols_x]):
        predicted_good_wine.append(i)
    for i in range(len(is_good_wine)):
        print(is_good_wine[i], predicted_good_wine[i])
        if is_good_wine[i] == predicted_good_wine[i] and predicted_good_wine[i] >= 5:
            srwg += 1
        elif is_good_wine[i] == predicted_good_wine[i] and predicted_good_wine[i] <= 5:
            srwb += 1
        elif is_good_wine[i] != predicted_good_wine[i] and predicted_good_wine[i] >= 5:
            sfwg += 1
        elif is_good_wine[i] != predicted_good_wine[i] and predicted_good_wine[i] <= 5:
            sfwb += 1
    df = pd.DataFrame([[srwg, srwb], [sfwg, sfwb]], index=['predict right', 'predict false'],
                      columns=['good wine', 'bad wine'])
    sns.heatmap(df, annot=True)
    plt.show()


def main():
    show_correlation_hitmap()
    classifier_testing(KNeighborsClassifier(), 'KNN')
    classifier_testing(GradientBoostingClassifier(), 'GB')
    classifier_testing(DecisionTreeClassifier(), 'Tree')
    classifier_testing(SVC(), 'SVM')
    classifier_testing(LogisticRegression(solver='lbfgs', max_iter=10000), 'LR')
    classifier_testing_comparison()
    predict_model()


if __name__ == '__main__':
    main()
