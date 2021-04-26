#!/usr/bin/env python
# coding: utf-8

from konlpy.tag import Mecab, Komoran
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

class tokenize_tfidf():
    def __init__(self):
        pass
    def tokenize_(self,x_data, x_data_h,y_data,y_data_h,tagger,tokenize_type):
        self.x_data=x_data
        self.x_data_h=x_data_h
        self.y_data=y_data
        self.y_data_h=y_data_h
        self.tagger=tagger

        #komoran_morphs
        x_data_komoran=[]
        for i in range(len(self.x_data)):
            x_data_komoran.append(tagger.morphs(x_data[i]))
        x_data_komoran_h=[]
        for i in range(len(x_data_h)):
            x_data_komoran_h.append(tagger.morphs(x_data_h[i]))

        # #komoran noun
        x_data_komoran_noun=[]
        x_data_komoran_noun_h=[]
        y_data_noun=[]
        y_data_noun_h=[]
        for i in range(len(x_data)):
            if len(tagger.nouns(x_data[i]))!=0:
                x_data_komoran_noun.append(tagger.nouns(x_data[i]))
                y_data_noun.append(y_data[i])
        for i in range(len(x_data_h)):
            if len(tagger.nouns(x_data_h[i]))!=0:
                x_data_komoran_noun_h.append(tagger.nouns(x_data_h[i]))
                y_data_noun_h.append(y_data_h[i])
        if tokenize_type=="m":
            return x_data_komoran,x_data_komoran_h
        elif tokenize_type=="n":
            return x_data_komoran_noun,x_data_komoran_noun_h,y_data_noun,y_data_noun_h
    def tokenize_for_kfold(self, x_data, y_data, tagger, tokenize_type):
        self.x_data = x_data
        self.y_data = y_data
        self.tagger = tagger

        # tagger_morphs
        x_data_komoran = []
        for i in range(len(self.x_data)):
            x_data_komoran.append(tagger.morphs(x_data[i]))

        # #tagger noun
        x_data_komoran_noun = []
        x_data_komoran_noun_h = []
        y_data_noun = []
        y_data_noun_h = []
        for i in range(len(x_data)):
            if len(tagger.nouns(x_data[i])) != 0:
                x_data_komoran_noun.append(tagger.nouns(x_data[i]))
                y_data_noun.append(y_data[i])
        # for i in range(len(x_data_h)):
        #     if len(tagger.nouns(x_data_h[i])) != 0:
        #         x_data_komoran_noun_h.append(tagger.nouns(x_data_h[i]))
        #         y_data_noun_h.append(y_data_h[i])
        if tokenize_type == "m":
            return x_data_komoran
        elif tokenize_type == "n":
            return x_data_komoran_noun, y_data_noun

    def tf_idf(self,x_data_komoran,x_data_komoran_h):
        self.x_data_komoran=x_data_komoran
        self.x_data_komoran_h=x_data_komoran_h

        x_data_komoran_sum=sum(x_data_komoran,[]) # word dictionary about traing data, so you need to one time make sum var

        tfidf_v1=TfidfVectorizer()
        x_train_komoran=[]
        tfidf_v1=tfidf_v1.fit(x_data_komoran_sum)

        #print(len(tfidf_v1.vocabulary_)) --> change 97 to 110 docause of tagger
        for index, data in enumerate(x_data_komoran):
            x = tfidf_v1.transform(data)
            x = x.toarray()
            xxxx=np.zeros(97,) # it is dtm shape
            for ind,data_ in enumerate(x):
                xxxx+=data_
            x_train_komoran.append(xxxx)

        # tfidf_v1_h=TfidfVectorizer()
        x_train_komoran_h=[]
        for index, data in enumerate(x_data_komoran_h):
            x = tfidf_v1.transform(data)
            x = x.toarray()
            xxxx=np.zeros(97,) # it is dtm shape
            for ind,data_ in enumerate(x):
                xxxx+=data_
            x_train_komoran_h.append(xxxx)

        x_train_komoran_arr=np.array(x_train_komoran)
        x_train_komoran_arr_h=np.array(x_train_komoran_h)

        return x_train_komoran_arr,x_train_komoran_arr_h
    def tf_idf_noun(self,x_data_komoran_noun,x_data_komoran_noun_h,y_data_noun,y_data_noun_h):
        tfidf_v2 = TfidfVectorizer()
        x_train_komoran_noun = []
        y_train_komoran_noun = []
        x_data_komoran_noun_sum = sum(x_data_komoran_noun, [])
        # print(x_data_komoran_noun[:5])
        tfidf_v2 = tfidf_v2.fit(x_data_komoran_noun_sum)
        # print(len(tfidf_v2.vocabulary_)) #this is dtm
        for index, data in enumerate(x_data_komoran_noun):
            x = tfidf_v2.transform(data)
            x = x.toarray()
            xxxx = np.zeros(38, )  # it is dtm shape
            for ind, data_ in enumerate(x):
                xxxx += data_
            x_train_komoran_noun.append(xxxx)
            y_train_komoran_noun.append(y_data_noun[index])
            # print(x_train_komoran_noun)
        x_train_komoran_h_noun = []
        y_train_komoran_h_noun = []
        for index, data in enumerate(x_data_komoran_noun_h):
            x = tfidf_v2.transform(data)
            x = x.toarray()
            xxxx = np.zeros(38, )  # it is dtm shape
            for ind, data_ in enumerate(x):
                xxxx += data_
            x_train_komoran_h_noun.append(xxxx)
            y_train_komoran_h_noun.append(y_data_noun_h[index])

        #change data type for ML model
        x_train_komoran_noun = np.array(x_train_komoran_noun)
        x_train_komoran_h_noun = np.array(x_train_komoran_h_noun)
        y_train_komoran_noun = pd.Series(y_train_komoran_noun)
        y_train_komoran_h_noun = pd.Series(y_train_komoran_h_noun)

        return x_train_komoran_noun, x_train_komoran_h_noun, y_train_komoran_noun, y_train_komoran_h_noun