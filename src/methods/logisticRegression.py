#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:58:49 2021

@author: matiasrolon
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, hamming_loss, coverage_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords


class MultilabelLogisticRegression:

    def __init__(self, settings):
        self.settings = settings
        self.models = dict()
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        stop_words = set(stopwords.words('english'))

        for category in self.settings.LABEL_COLUMNS:
            print('... Training category {}'.format(category))
            LogReg_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])
            LogReg_pipeline.fit(x_train, y_train[category])
            self.models[category] = LogReg_pipeline

    def predict(self, x_test, y_test):
        self.y_test = y_test
        self.x_test = x_test
        metrics_per_label = dict()
        prediction_matrix = []

        for category in self.settings.LABEL_COLUMNS:
            print('... Predicting category {}'.format(category))
            prediction = self.models[category].predict(x_test)
            prediction_matrix.append(prediction)
            # calculate metrics
            accuracy = accuracy_score(y_test[category], prediction)
            print("accuracy", accuracy)
            fpr, tpr, _ = roc_curve(y_test[category], prediction)
            roc_auc = auc(fpr, tpr)
            loss = hamming_loss(y_test[category], prediction)

            metrics_per_label[category] = dict(
                accuracy=accuracy,
                roc=dict(
                    fpr=fpr,
                    tpr=tpr,
                    roc_auc=roc_auc
                ),
                hamming_loss=loss
            )

        prediction_matrix = np.array(prediction_matrix)
        return prediction_matrix, metrics_per_label

    def accuracy_total(self, y_test, y_pred):
        return accuracy_score(y_test, y_pred)

    def coverage_error_total(self, y_test, y_pred):
        return coverage_error(y_test, y_pred)

    def graph_roc_curves(self, metrics):
        # graficar
        plt.figure()
        lw = 2

        colors = ["darkorange", "aqua", "azure", "green", "indigo", "lavender", "lime", "magenta", "olive", "orange", "pink", "purple",
                  "red", "salmon", "silver", "teal", "yellow", "coral", "ivory", "gold", "orchid", "tan", "fuchsia", "darkgreen",
                  "brown", "navy", "black", "cyan"]
        index = 0
        for emotion in metrics:
            if metrics[emotion]["roc"]["roc_auc"]>=0.5:
                plt.plot(
                    metrics[emotion]["roc"]["fpr"],
                    metrics[emotion]["roc"]["tpr"],
                    color=colors[index],
                    lw=lw,
                    label=emotion + " (area ="+ str(metrics[emotion]["roc"]["roc_auc"]) +")"
                )
            index+=1

        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.show()
