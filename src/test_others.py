#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:58:49 2021

@author: matiasrolon
"""

# Libraries y clases
import os.path
import pandas as pd

from methods.logisticRegression import MultilabelLogisticRegression
from methods.KNearestNeighbours import MultilabelKNearestNeighbours
import settings
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


if __name__ == "__main__":

    data= None
    data_train = None
    data_test = None
    stop_words = set(stopwords.words('english'))
    metrics = []

    print("load dataset ----------------")
    if settings.PREPROCESS_DATA:                                        # Verifica si se debe preprocesar el dataset segun config.
        print("preproccessing ----------------")
        pp = Preprocessor(settings.PATH_FOLDER_DATA)
        data_train = pp.preproccess(settings.PATH_TRAIN, settings.PATH_EMOTIONS, "train")
        data_test = pp.preproccess(settings.PATH_TEST, settings.PATH_EMOTIONS, "test")
        data = pd.concat([data_train, data_test])
        pp.write(data, settings.PATH_PREPROCESSING_DATA)
    else:
        data = pd.read_csv(settings.PATH_PREPROCESSING_DATA)            # Extraigo data de archivo guardado con anterioridad.

    data = data.iloc[: , 1:]                                            # Elimino la primer columna sin nombre (son los ids)
    data['comment'] = data['comment'].map(lambda com: clean_text(com))
    data_train = data[data.tag == "train"]
    data_test = data[data.tag == "test"]
    x_train = data_train.comment
    y_train = data_train.drop(labels=['tag','comment','emotions'], axis=1)
    x_test = data_test.comment
    y_test = data_test.drop(labels=['tag','comment','emotions'], axis=1)

    # print("******************************************************************************************")
    # print("REGRESION LOGISTICA **********************************************************************")
    # print("******************************************************************************************")
    # logReg = MultilabelLogisticRegression(settings)
    # logReg.train(x_train, y_train)
    # predictions, reg_metrics = logReg.predict(x_test, y_test)
    # metrics.append(reg_metrics)
    # print("METRICS ", reg_metrics)
    # transpose_predictions = predictions.T
    # np_y_test = np.array(y_test)
    #
    # print("- accuracy: ", logReg.accuracy_total(np_y_test, transpose_predictions))
    # print("- coverage_error: ", logReg.coverage_error_total(np_y_test, transpose_predictions))
    #
    # logReg.graph_roc_curves(reg_metrics)
    #
    # print("******************************************************************************************")
    # print("K NEAREST NEIGHBOURS *********************************************************************")
    # print("******************************************************************************************")
    # KNeighbours = MultilabelKNearestNeighbours(settings)
    # KNeighbours.train(x_train, y_train)
    # predictions, knn_metrics = KNeighbours.predict(x_test, y_test)
    # metrics.append(knn_metrics)
    # print("GENERAL METRICS ", knn_metrics)
    # transpose_predictions = predictions.T
    # np_y_test = np.array(y_test)
    #
    # print("- accuracy: ", KNeighbours.accuracy_total(np_y_test, transpose_predictions))
    # print("- coverage_error: ", KNeighbours.coverage_error_total(np_y_test, transpose_predictions))
    #
    # KNeighbours.graph_roc_curves(knn_metrics)
    #
    print("******************************************************************************************")
    print("NEURAL NETWORKS - ************************************************************************")
    print("******************************************************************************************")

    # Configuration options
    n_samples = data.shape[0]
    n_features = 28
    n_classes = 3
    n_epochs = 50
    random_state = 42
    batch_size = 250
    verbosity = 1
    validation_split = 0.2

    # Split into training and testing data
    X_data = data["comment"]
    y_data = data.drop(labels=['tag', 'comment', 'emotions'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.20, random_state=random_state)

    # Convert text to embedded vectors
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data.comment)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 200
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    # Create the model
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=maxlen))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(n_features, activation='sigmoid'))

    AUC_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC',
        summation_method='interpolation', name=None, dtype=None,
        thresholds=None, multi_label=True, num_labels=28, label_weights=None,
        from_logits=False
    )

    # Compile the model
    model.compile(loss=binary_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy', AUC_metric])


    # Fit data to model
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              verbose=verbosity)

    # Generate generalization metrics
    score = model.evaluate(X_test, y_test, verbose=0)
    print("model metrics", model.metrics_names)

    print(f'Test loss: {score[0]} / Test accuracy: {score[1]} / test Auc: {score[2]}')

