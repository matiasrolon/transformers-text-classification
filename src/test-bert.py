#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:58:49 2021

@author: matiasrolon
"""

# Libraries y clases
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib import rc
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from preprocessing import Preprocessor
from commentsDataset import CommentsDataset
from commentsDataModule import CommentsDataModule
from commentTagger import CommentsTagger
import settings

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import seaborn as sns


if __name__ == "__main__":

    data= None
    data_train = None
    data_test = None

    if settings.PREPROCESS_DATA:
        pp = Preprocessor(settings.PATH_FOLDER_DATA)
        data_train = pp.proproccess(settings.PATH_TRAIN, settings.PATH_EMOTIONS, "train")
        data_test = pp.proproccess(settings.PATH_TEST, settings.PATH_EMOTIONS, "test")
        data = pd.concat([data_train, data_test])
        pp.write(data, settings.PATH_PREPROCESSING_DATA)
    else:
        data = pd.read_csv(settings.PATH_PREPROCESSING_DATA)            # Extraigo data de archivo guardado con anterioridad.

    data = data.iloc[: , 1:]                                            # Elimino la primer columna sin nombre (son los ids)
    data_train = data[data.tag == "train"].drop(columns="tag")
    data_test = data[data.tag == "test"].drop(columns="tag")

    tokenizer = BertTokenizer.from_pretrained(settings.BERT_MODEL_NAME)

    # EJEMPLO DE DATASET TOKENIZADO
    """
    train_dataset = CommentsDataset(
        data_train,
        tokenizer,
        max_token_len=settings.MAX_TOKEN_COUNT
    )
    sample_item = train_dataset[0]
    print(sample_item.keys())
    print(sample_item["comment_text"])
    print(sample_item["labels"])
    print(sample_item["input_ids"].shape)
    """

    # EJEMPLO DE DATASET CONVERTIDO EN DATA MODULE  (PYTHON LIGHNING)

    data_module = CommentsDataModule(
        data_train,
        data_test,
        tokenizer,
        batch_size=settings.BATCH_SIZE,
        max_token_len=settings.MAX_TOKEN_COUNT
    )

    steps_per_epoch = len(data_train) // settings.BATCH_SIZE
    total_training_steps = steps_per_epoch * settings.N_EPOCHS
    warmup_steps = total_training_steps // 5

    model = CommentsTagger(
        n_classes=len(settings.LABEL_COLUMNS),
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps
    )

    """ESTADISTICAS"""
    # Frecuencia por palabra (grafico)
    """
    data[columns_names].sum().sort_values().plot(kind="barh")
    plt.show()
    """

    # Cantidad de tokens por palabra
    """
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    token_count_arr = []
    for _, row in train_df.iterrows():
      token_count = len(tokenizer.encode(
        row["comment"],
        max_length=MAX_TOKEN_COUNT,
        truncation=True
      ))
      token_count_arr.append(token_count)
    sns.histplot(token_count_arr)
    plt.xlim([0, 100]);
    plt.show()
    """


    # Convierto valores de columnas de emociones en un array.
    #data['emotions'] = data[data.columns[2:]].values.tolist()
    #data = data[['comment', 'emotions']]
    #print(data.head())