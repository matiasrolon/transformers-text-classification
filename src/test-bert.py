#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:58:49 2021

@author: matiasrolon
"""

# Libraries y clases
import pandas as pd
import numpy as np
from preprocessing import Preprocessor


# Constants
PATH_FOLDER_DATA = "../dataset/multilabel-emotions-clasifications"
PATH_TEST = PATH_FOLDER_DATA+ "/test.tsv"
PATH_TRAIN = PATH_FOLDER_DATA + "/train.tsv"
PATH_EMOTIONS = PATH_FOLDER_DATA + "/emotions.txt"
NAME_PREPROCESSING_DATA = 'preprocess_emotions.csv'

if __name__ == "__main__":
    pp = Preprocessor(PATH_FOLDER_DATA)
    data_train = pp.proproccess(PATH_TRAIN, PATH_EMOTIONS, "train")
    data_test = pp.proproccess(PATH_TEST, PATH_EMOTIONS, "test")

    data = pd.concat([data_train, data_test])
    print(data)
    pp.write(data, NAME_PREPROCESSING_DATA)