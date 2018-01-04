from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
import tools.data_loader as dl
import sys
import numpy as np


def rand_cv_optimized_svm(kernel, limit):
    #Build Hyperspace for Parameter search.
    Cs = np.logspace(-10, 10, 150)
    gamma = np.logspace(-15, 15, 150)

    #set up parameterless svm
    svc = svm.SVC(kernel=kernel)

    #load train and test set
    data_train, labels_train = dl.DataLoader.load('train.csv', limit=None)
    data_test, labels_test = dl.DataLoader.load('test.csv', limit=None)

    #Cross validation of 30 of the Parameters in the Hyperspace on the SVM
    clf = RandomizedSearchCV(estimator=svc, param_distributions={"C": Cs, "gamma": gamma}, n_iter=30)
    clf.fit(X=data_train, y=labels_train)

    #Predicting the test set according to the best svm given by the Randomized Parameter Search
    result = clf.best_estimator_.predict(X=data_test)

    # Accuracy of SVM prediciton
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

    print(rand_cv_optimized_svm(kernel, limit))