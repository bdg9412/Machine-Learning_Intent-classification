import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import pickle #for model save
import joblib #for model save

class ml_classification():
    def __init__(self):
        pass
    def do_svm(self,x_train_komoran_arr,x_train_komoran_h,y_data,y_data_h):

        # tuned_parameters = {
        #     'C': (np.arange(0.1, 20, 0.1)), 'kernel': ['linear'],
        #     'C': (np.arange(0.1, 20, 0.1)), 'gamma':(np.arange(0.1, 20, 0.1)), 'kernel': ['rbf'],
        #     'degree': [2, 3, 4], 'gamma': (np.arange(0.1, 20, 0.1)), 'C': (np.arange(0.1, 20, 0.1)),
        #     'kernel': ['poly']
        # }
        # svm_model = SVC()
        # model_svm = GridSearchCV(svm_model, tuned_parameters, cv=10, scoring='accuracy')
        # model_svm.fit(x_train_komoran_arr, y_data)
        # y_pred_h = model_svm.predict(x_train_komoran_h)
        # str_re = "SVM acc: " + str(accuracy_score(y_pred_h, y_data_h))
        # acc_svm = accuracy_score(y_pred_h, y_data_h)
        # return str_re, acc_svm, model_svm
        iteration=np.arange(0.1,50,0.1)
        kernel_list=['rbf','linear','poly','sigmoid']
        acc_svm=0
        for index,data_ in enumerate(kernel_list):
            for i in iteration:
                svm_model = SVC(kernel=data_, C=i, gamma='auto', probability=True, random_state=0)
                svm_model.fit(x_train_komoran_arr, y_data)
                y_pred_h = svm_model.predict(x_train_komoran_h)
                if acc_svm<accuracy_score(y_pred_h, y_data_h):
                    svm_model_dost=svm_model
                    str_re="SVM acc: "+str(accuracy_score(y_pred_h, y_data_h))
                    acc_svm=accuracy_score(y_pred_h, y_data_h)
        #print(str_re)
        joblib.dump(svm_model_dost, './svm_dost_first.pkl')
        return str_re,acc_svm,svm_model_dost,


    def do_knn(self,x_train_komoran_arr,x_train_komoran_h,y_data,y_data_h):
        acc_knn=0
        for i in range(10):
            neigh = KNeighborsClassifier(n_neighbors=i + 1)
            neigh.fit(x_train_komoran_arr, y_data)
            result_knn = neigh.predict(x_train_komoran_h)
            acc = accuracy_score(result_knn, y_data_h)
            if acc_knn<acc:
                acc_knn=acc
                where_var=i
                neigh_dost=neigh
        print("KNN acc dost: ",acc_knn," when neighbors: ",where_var)
        str_re="KNN acc dost: "+str(acc_knn)+" when neighbors: "+str(where_var)
        return str_re,acc_knn,neigh_dost

    def do_nb(self,x_train_komoran_arr,x_train_komoran_arr_h,y_data,y_data_h):
        gnb = GaussianNB()
        gnb.fit(x_train_komoran_arr, y_data)
        result_nb = gnb.predict(x_train_komoran_arr_h)

        print("Naive bayes acc: ",accuracy_score(result_nb, y_data_h))
        str_re = "Naive bayes acc: "+ str(accuracy_score(result_nb, y_data_h))
        conf=accuracy_score(result_nb, y_data_h)
        return str_re,conf,gnb

    def do_svm_2(self,x_train_komoran_arr,y_data):

        tuned_parameters = {
            'C': (np.arange(0.1, 20, 0.1)), 'kernel': ['linear'],
            'C': (np.arange(0.1, 20, 0.1)), 'gamma':(np.arange(0.1, 20, 0.1)), 'kernel': ['rbf'],
            'degree': [2, 3, 4], 'gamma': (np.arange(0.1, 20, 0.1)), 'C': (np.arange(0.1, 20, 0.1)),
            'kernel': ['poly']
        }
        svm_model = SVC()
        model_svm = GridSearchCV(svm_model, tuned_parameters, cv=10, scoring='accuracy')
        model_svm.fit(x_train_komoran_arr, y_data)
        return model_svm