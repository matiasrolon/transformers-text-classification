#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:58:49 2021

@author: matiasrolon
"""

# Libraries y clases
import os.path
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib import rc
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from preprocessor import Preprocessor
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

    if not os.path.isfile(settings.PATH_TRAINED_MODEL):         # Si el modelo entrenado no existe.
        print("training------------------------")
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )
        logger = TensorBoardLogger("lightning_logs", name="reddit_comments")  # Logea el progreso del entrenamiento
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
        trainer = pl.Trainer(
            logger=logger,
            checkpoint_callback=True,
            callbacks=[early_stopping_callback],
            max_epochs=settings.N_EPOCHS,
            gpus=1,
            progress_bar_refresh_rate=30
        )
        trainer.fit(model, data_module)                                     # Entrena el modelo.
        trainer.save_checkpoint(settings.PATH_TRAINED_MODEL)

    print("loading trained model ------------")
    trained_model = CommentsTagger.load_from_checkpoint(
        settings.PATH_TRAINED_MODEL,
        n_classes=len(settings.LABEL_COLUMNS)
    )

    with torch.no_grad():
        print("eval----------------------------")
        print(trained_model.eval())
        print(trained_model.freeze())


    print("evaluation----------------------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = trained_model.to(device)
    test_dataset = CommentsDataset(
        data_test,
        tokenizer,
        max_token_len=settings.MAX_TOKEN_COUNT
    )
    predictions = []
    labels = []
    for item in tqdm(test_dataset):
        _, prediction = trained_model(
            item["input_ids"].unsqueeze(dim=0).to(device),
            item["attention_mask"].unsqueeze(dim=0).to(device)
        )
        predictions.append(prediction.flatten())
        labels.append(item["labels"].int())
    predictions = torch.stack(predictions).detach().cpu()
    labels = torch.stack(labels).detach().cpu()

    print("metrics -------------------------")
    print("- Accuracy")
    print(accuracy(predictions, labels, threshold=settings.THRESHOLD))
    print("")
    print("- AUROC per emotion")
    for i, name in enumerate(settings.LABEL_COLUMNS):
      emotion_auroc = auroc(predictions[:, i], labels[:, i], pos_label=1)
      print(f"{name}: {emotion_auroc}")