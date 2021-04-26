#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Mecab, Komoran

class ready_for_classification():
    def __init__(self):
        self.flag=0
    def preparation(self,flag):
        self.flag=flag
        data = pd.read_csv('./something.csv',encoding='utf-8') #csv data load
        data_h = pd.read_csv('./something_h.csv',encoding='utf-8') #csv data load
        print(data.head()) #show data top 5
        # print(data.shape)
        # print(data_h.shape)

        data_list=data.values.tolist() #pandas to list
        data_list_h=data_h.values.tolist() #pandas to list

        #choice the tokenizer
        # flag = 1
        if flag == 0:
            tagger = Komoran()
        elif flag == 1:
            tagger = Mecab()

        x_data=data['content'] #assign data with column
        y_data=data['intent'] #assign data with column
        x_data_h=data_h['content'] #assign data with column
        y_data_h=data_h['intent'] #assign data with column

        return x_data, x_data_h,y_data,y_data_h,tagger
        #return x_data, y_data, tagger #kfold