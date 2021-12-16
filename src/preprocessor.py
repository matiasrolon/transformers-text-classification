#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:58:49 2021

@author: matiasrolon
"""

# Libraries
import pandas as pd
import numpy as np


class Preprocessor:

    def __init__(self, path_folder="./"):
        self.path_folder = path_folder              # Carpeta base donde se encuentran los datos, graban archivos nuevos, etc.

    def write(self, data, path_write='./preprocessing.csv'):
        if isinstance(data, pd.DataFrame):          # Si es un dataframe.
            data.to_csv(path_write)                 # Grabo archivo en path.

    def preproccess(self, path_data,  path_emotions="./emotions.txt", tag=None):
        print("Preprocesamiento de dataset en ", path_data)

        # Carga data
        data = pd.read_csv(
            path_data,
            sep='\t',
            names=["comment", "emotions", "code"]
        )
        data = data.drop(columns="code")
        if tag is not None:
            data["tag"] = tag

        # Carga emociones
        emotions = []
        with open(path_emotions) as file_emotions:
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
