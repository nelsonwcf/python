# -*- coding: utf-8 -*-
"""Chapter 3"""

import ssl
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# workaround for the code to work in CHARTER Environment
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # the effective code beigns here
    def show_img(digit):
        """Plots on digit on the screen"""
        img = digit.reshape(28, 28)
        plt.imshow(img, cmap=matplotlib.cm.binary, interpolation='nearest')
        plt.axis("off")
        plt.show()

    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
        """Plots precision and recall vs threshold"""
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0, 1])
        plt.xlim([0, 1])

    def plot_precision_vs_recall(precisions, recalls):
        """Plots precision vs recall"""
        plt.plot(recalls, precisions)
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.xlabel("Recall")
        plt.ylabel("Precision")

    def plot_roc_curve(fpr, tpr, label=None):
        """Plots ROC curve"""
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

    def plot_digits(instances, images_per_row=10, **options):
        """Plots groups of digits"""
        size = 28
        images_per_row = min(len(instances), images_per_row)
        images = [instance.reshape(size, size) for instance in instances]
        n_rows = (len(instances) - 1) // images_per_row + 1
        row_images = []
        n_empty = n_rows * images_per_row - len(instances)
        images.append(np.zeros((size, size * n_empty)))
        for row in range(n_rows):
            rimages = images[row * images_per_row : (row + 1) * images_per_row]
            row_images.append(np.concatenate(rimages, axis=1))
        image = np.concatenate(row_images, axis=0)
        plt.imshow(image, cmap=matplotlib.cm.binary, **options)
        plt.axis("off")

    # fetching and segmenting the dataset
    X, y = fetch_openml('mnist_784', return_X_y=True) # mldata is deprecated
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # shuffling the inputs
    shuffled_index = np.random.permutation(60000)
    X_train = X_train[shuffled_index]
    y_train = y_train[shuffled_index]

    # testing a binary classifier
    # convert the categories ('1','2',...,'9','0') to true ('5') or false
    y_train_5 = (y_train == '5')
    y_test_5 = (y_test == '5')

    # creating and training a classifier
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)

    # scoring the classifier
    print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
    y_pred = cross_val_predict(sgd_clf, X_train, y_train_5)

    # plotting results in confusion matrix
    print(confusion_matrix(y_train_5, y_pred))

    # getting precision and recall
    precision = precision_score(y_train_5, y_pred) # how many correctly classified
    recall = recall_score(y_train_5, y_pred) # how many positives were detected
    f1 = f1_score(y_train_5, y_pred)

    # getting threshold from clasifier instead of the prediction
    print(sgd_clf.decision_function(X[10000].reshape(1, -1))) # get score for class

    # customizing threshold to tune recall and precison
    y_scores = cross_val_predict(sgd_clf,
                                 X_train,
                                 y_train_5,
                                 method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    y_train_5_ht = y_scores > -200000
    print(precision_score(y_train_5, y_train_5_ht),
          recall_score(y_train_5, y_train_5_ht))
    
    # ROC curve
    fpr, tpr, thresholds, = roc_curve(y_train_5, y_scores)
    plot_roc_curve(fpr, tpr)
    print(roc_auc_score(y_train_5, y_scores))
    
    # Comparison between SGD and RandomForest classifiers using ROC metrics
    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_clf,
                                        X_train,
                                        y_train_5,
                                        cv=3,
                                        method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1] # get the positive class probability
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,
                                                          y_scores_forest)
    
    # plot comparison between SGD and RandomForests
    plt.plot(fpr, tpr, "b:", label="SGD")
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
    some_digit_scores = sgd_clf.decision_function(X[1000].reshape(1, -1)) # '0'
    np.argmax(some_digit_scores) # returns the index with max element
    sgd_clf.classes_ # compare it to the classes
    
    # force classifiers to use OvO or OvR
    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
    ovo_clf.fit(X_train, y_train)
    ovo_clf.predict(X[1000].reshape(1, -1))
    len(ovo_clf.estimators_) # get how many classifiers were trained
    
    # Using RandomForests, which is already a multiclass classifier natively
    forest_clf.fit(X_train, y_train)
    forest_clf.predict(X[1000].reshape(1, -1))
    forest_clf.predict_proba(X[1000].reshape(1, -1))
    forest_clf.classes_
    
    # Improving results using a Scaler (why does it work since they are images)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    cross_val_score(sgd_clf, X_train_scaled, y_train, cv=5, scoring="accuracy")
    
    # Error Analysis
    sgd_clf.fit(X_train_scaled, y_train)
    y_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_pred)
    
    # Make the confusion matrix graphical
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()
    
    # Convert from absolute values to percentages
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums # review python vector operations
    
    # Plot only the errors
    np.fill_diagonal(norm_conf_mx, 0) # numpy really has some obscure functions!
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    
    # Plotting individual errors
    cl_a, cl_b = '3', '5'
    X_aa = X_train[(y_train == cl_a) & (y_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_pred == cl_b)]
    
    plt.figure(figsize=(8, 8))
    plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
    plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
    plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
    plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
    plt.show()
    
    # Multilabel Classification
    y_train_large = (y_train.astype(int) >= 7)
    y_train_odd = (y_train.astype(int) % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]
    
    # Multilabel Classifier KNeighbors
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)
    knn_clf.predict(X[1000].reshape(1, -1))
    
    # Computing a metric
    # Note, using smaller sample due to the time it takes for knn
    y_train_knn_pred = cross_val_predict(knn_clf, X_train[:1000], y_multilabel[:1000], cv=3)
    f1_score(y_multilabel[:1000], y_train_knn_pred, average='macro')
    
    # Multioutput Multiclass Classification
    noise = np.random.randint(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise
    y_train_mod = X_train
    y_test_mod = X_test
    knn_clf.fit(X_train_mod, y_train_mod)
    
    # checking cleaning-up results
    img_id = 0
    clean_digit = knn_clf.predict(X_test_mod[img_id].reshape(1, -1))
    plt.figure(figsize=(8, 8))
    plt.subplot(311); show_img(X_test_mod[img_id])
    plt.subplot(312); show_img(y_test_mod[img_id])
    plt.subplot(313); show_img(clean_digit)
    plt.show()

if __name__ == "__main__":
    main()
