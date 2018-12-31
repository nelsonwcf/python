# -*- coding: utf-8 -*-

# workaround for the code to work in CHARTER Environment
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# the effective code beigns here
def show_img(digit):
    import matplotlib
    import matplotlib.pyplot as plt
    
    img = digit.reshape(28,28)
    plt.imshow(img, cmap = matplotlib.cm.binary, interpolation='nearest')
    plt.axis("off")
    plt.show()
    
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    import matplotlib.pyplot as plt

    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label = "Recall")    
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
    plt.xlim([0,1])
    
def plot_precision_vs_recall(precisions, recalls):
    import matplotlib.pyplot as plt

    plt.plot(recalls, precisions)
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    
# fetching and segmenting the dataset
from sklearn.datasets import fetch_openml
X,y = fetch_openml('mnist_784', return_X_y=True) # mldata is deprecated
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# shuffling the inputs
import numpy as np
shuffled_index = np.random.permutation(60000)
X_train = X_train[shuffled_index]
y_train = y_train[shuffled_index]

# testing a binary classifier
# convert the categories ('1','2',...,'9','0') to true ('5') or false
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

# creating and training a classifier 
from sklearn.linear_model import SGDClassifier
sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, y_train_5)

# scoring the classifier
from sklearn.model_selection import cross_val_score, cross_val_predict
print(cross_val_score(sgd_classifier, X_train, y_train_5, cv = 3, scoring="accuracy"))
y_predicted = cross_val_predict(sgd_classifier, X_train, y_train_5)

# plotting results in confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5, y_predicted))

# getting precision and recall
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_train_5, y_predicted) # of the detected, how many were correctly classified
recall = recall_score(y_train_5, y_predicted) # how many of the correct ones were detected
f1 = f1_score(y_train_5, y_predicted)

# getting threshold from clasifier instead of the prediction
print(sgd_classifier.decision_function(X[10000].reshape(1,-1))) # get score instead of classification

# customizing threshold to tune recall and precison
from sklearn.metrics import precision_recall_curve
y_predicted_scores = y_predicted = cross_val_predict(sgd_classifier, X_train, y_train_5, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_predicted_scores)
y_train_5_ht = y_predicted_scores > -200000
print(precision_score(y_train_5, y_train_5_ht),recall_score(y_train_5, y_train_5_ht))
