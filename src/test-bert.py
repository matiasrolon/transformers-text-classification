#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:58:49 2021

@author: matiasrolon
"""

# Libraries
import pandas as pd
import numpy as np


# Constants
PATH_FOLDER_DATA = "../dataset/multilabel-emotions-clasifications"
PATH_TEST = PATH_FOLDER_DATA+ "/test.tsv"
PATH_TRAIN = PATH_FOLDER_DATA + "/train.tsv"
PATH_EMOTIONS = PATH_FOLDER_DATA + "/emotions.txt"


def proproccess_dataset(path_file, tag="undefined"):
    # Carga data
    data = pd.read_csv(
        path_file,
        sep='\t',
        names=["comment", "emotions", "code"]
    )
    data = data.drop(columns="code")
    data["tag"] = tag

    # Carga emociones
    emotions = []
    with open(PATH_EMOTIONS) as file_emotions:
        lines_emotions = file_emotions.readlines()
        for line in lines_emotions:                         # Setea el nombre de todas las emociones posibles en un array
            column_name = line.rstrip()
            emotions.append(column_name)
            data[column_name] = 0

    for index, row in data.iterrows():
        row_emotion_ids = row["emotions"].split(',')        # Extrae las emociones de ese comentario
        for emotion_id in row_emotion_ids:
            column_name = emotions[int(emotion_id)]
            data.loc[index, column_name] = 1                # Setea con 1 aquellas emociones que estan en el comentario.

    return data


if __name__ == "__main__":
    data_train = proproccess_dataset(PATH_TRAIN, "train")
    data_test = proproccess_dataset(PATH_TEST, "test")

    data = pd.concat([data_train, data_test])
    print(data)
    data.to_csv('preprocess_emotions.csv')