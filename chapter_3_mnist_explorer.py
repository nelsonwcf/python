# -*- coding: utf-8 -*-

# workaround for the code to work in CHARTER Environment
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# the effective code beigns here
import matplotlib
import matplotlib.pyplot as plt

def show_img(digit):
    img = digit.reshape(28,28)
    plt.imshow(img, cmap = matplotlib.cm.binary, interpolation='nearest')
    plt.axis("off")
    plt.show()
    
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label = "Recall")    
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
    plt.xlim([0,1])
    
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions)
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")

def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
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
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# scoring the classifier
from sklearn.model_selection import cross_val_score, cross_val_predict
print(cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring="accuracy"))
y_pred = cross_val_predict(sgd_clf, X_train, y_train_5)

# plotting results in confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5, y_pred))

# getting precision and recall
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_train_5, y_pred) # how many correctly classified
recall = recall_score(y_train_5, y_pred) # how many positives were detected
f1 = f1_score(y_train_5, y_pred)

# getting threshold from clasifier instead of the prediction
print(sgd_clf.decision_function(X[10000].reshape(1,-1))) # get score for class

# customizing threshold to tune recall and precison
from sklearn.metrics import precision_recall_curve
y_scores = cross_val_predict(sgd_clf,
                             X_train,
                             y_train_5,
                             method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
y_train_5_ht = y_scores > -200000
print(precision_score(y_train_5, y_train_5_ht),
      recall_score(y_train_5, y_train_5_ht))

# ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds, = roc_curve(y_train_5, y_scores)
plot_roc_curve(fpr, tpr)
print(roc_auc_score(y_train_5, y_scores))

# Comparison between SGD and RandomForest classifiers using ROC metrics
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf,
                                    X_train,
                                    y_train_5, 
                                    cv = 3,
                                    method="predict_proba")
y_scores_forest = y_probas_forest[:,1] # get the positive class probability
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,
                                                      y_scores_forest)

# plot comparison between SGD and RandomForests
plt.plot(fpr,tpr,"b:",label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

print(roc_auc_score(y_train_5, y_scores),
      roc_auc_score(y_train_5, y_scores_forest))

# checking recall and scores of the new model
y_pred_forest = y_scores_forest > 0.5
precision_forest = precision_score(y_train_5, y_pred_forest) 
recall_forest = recall_score(y_train_5, y_pred_forest) 
f1_forest = f1_score(y_train_5, y_pred_forest)

# MULTICLASS Classification
# using regular classifiers (which use OvO or OvR strategies)
sgd_clf.fit(X_train, y_train)
some_digit_scores = sgd_clf.decision_function(X[1000].reshape(1,-1)) # '0'
np.argmax(some_digit_scores) # returns the index with max element
sgd_clf.classes_ # compare it to the classes

# force classifiers to use OvO or OvR
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict(X[1000].reshape(1,-1))
len(ovo_clf.estimators_) # get how many classifiers were trained

# Using RandomForests, which is already a multiclass classifier natively
forest_clf.fit(X_train, y_train)
forest_clf.predict(X[1000].reshape(1,-1))
forest_clf.predict_proba(X[1000].reshape(1,-1))
forest_clf.classes_

# Improving results using a Scaler (why does it work since they are images)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv = 5, scoring = "accuracy")