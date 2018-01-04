from sklearn.model_selection import GridSearchCV
from sklearn import svm
import tools.data_loader as dl
import numpy as np
import sys


def grid_cv_optimized_svm(kernel, limit):

    # Build Hyperspace for Parameter search.
    Cs = np.logspace(-6, 9, 15)
    gamma = np.logspace(-15, 3, 15)

    #Initializing Parameterless svm
    svc = svm.SVC(kernel=kernel)

    #Load the train and test dataset
    data_train, labels_train = dl.DataLoader.load('train.csv', limit=limit)
    data_test, labels_test = dl.DataLoader.load('test.csv', limit=limit)

    #Extensive Search of the Paramater Hyperspace for the C and gamma parameter.
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs, gamma=gamma), n_jobs=1)
    clf.fit(X=data_train, y=labels_train)

    #Predict tets set according to best svm given by the GridSearch
    result = clf.best_estimator_.predict(X=data_test)

    # Accuracy of SVM Prediction
    correct = 0
    for i, label in enumerate(labels_test):
     if label == result[i]:
            correct += 1
    accuracy = (correct / len(data_test))

    return accuracy, clf.best_estimator_, clf.best_params_, clf.best_score_

# Execute
if len(sys.argv) < 2:
    print('Usage: python svm.py <KERNEL> [LIMIT]\n'
          '\tKERNEL: "linear" or "poly"\n'
          '\tLIMIT: any integer')

else:
    kernel = sys.argv[1]

    limit = None
    if len(sys.argv) == 3:
        limit = int(sys.argv[2])

    print(grid_cv_optimized_svm(kernel, limit))