from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
import tools.data_loader as dl
import numpy as np

# Parameters
# Number of neurons in the hidden layer, range: [10, 100]
# Number of hidden layers: length + 2

# Learning rate c, range [0.1, 1]
# c = 0.5
# n = Number of hidden nodes in nn, needs to be in format (x,)
n = list()
for i in range(50, 70):
    n.append((i,))
    i += 2
cs = np.linspace(0.1, 1, 10)

#setup of sckit learn Classifier
mlp = MLPClassifier(solver='lbfgs')

data_train, labels_train = dl.DataLoader.load('train.csv', limit=1000)
data_test, labels_test = dl.DataLoader.load('test.csv', limit=1000)

# faster, heuristic method.
clf = RandomizedSearchCV(estimator=mlp, param_distributions={"hidden_layer_sizes": n, "learning_rate_init": cs,
                                                             "max_iter": np.linspace(200, 250, 10)}, n_iter=100)

#Complete search of Hyperparameterspace, over hidden_layers, learning rate and max_iter.

#clf = GridSearchCV(estimator=mlp, param_grid=dict(hidden_layer_sizes=n, learning_rate_init=cs,
#                                                  max_iter=np.linspace(200, 250, 10)))

#actual training of the model given the data
clf.fit(X=data_train, y=labels_train)

print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)

result = clf.best_estimator_.predict(X=data_test)

# Accuracy
correct = 0
for i, label in enumerate(labels_test):
    if label == result[i]:
        correct += 1

accuracy = correct / len(data_test)
print(accuracy)
