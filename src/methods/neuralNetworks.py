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
from sklearn.neighbors import KNeighborsClassifier

from nltk.corpus import stopwords


class MultilabelKNearestNeighbours:

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
            print('Training category {}...'.format(category))
            MLkNN_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, weights='uniform'), n_jobs=1)),
            ])
            MLkNN_pipeline.fit(x_train, y_train[category])
            self.models[category] = MLkNN_pipeline

    def predict(self, x_test, y_test):
        self.y_test = y_test
        self.x_test = x_test
        metrics_per_label = dict()
        prediction_matrix = []
        print('Predicting categories...')
        for category in self.settings.LABEL_COLUMNS:
            prediction = self.models[category].predict(x_test)
            prediction_matrix.append(prediction)
            # calculate metrics
            accuracy = accuracy_score(y_test[category], prediction)
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