#!/usr/bin/frenv python
# coding: utf-8

import do_preparations
import do_tokenize
import do_classification

def do_init(tokenize_type):
    #Preparations data for classification
    pre=do_preparations.ready_for_classification()
    x_data, x_data_h,y_data,y_data_h,tagger=pre.preparation(0)
    # x_data, y_data,  tagger = pre.preparation(1) #kfold

    #choice tokenizet type
    tokenize_type=tokenize_type #if m than morphs or n than nouns

    #Tokenize and TF-IDF and Classification
    token=do_tokenize.tokenize_tfidf()
    if tokenize_type=="m":
        x_data_tagger,x_data_tagger_h=token.tokenize_(x_data, x_data_h,y_data,y_data_h,tagger,tokenize_type) #tokenize
        # x_data_tagger = token.tokenize_(x_data, y_data, tagger, #kfold
        #                                                tokenize_type)  # tokenize

        x_train_tagger_arr, x_train_tagger_arr_h = token.tf_idf(x_data_tagger, x_data_tagger_h)  # tf-idf
        # x_train_tagger_arr = token.tf_idf(x_data_tagger) #kfold

        # #k_fold
        #
        # kfold.kfold_do(x_train_tagger_arr,y_data)

        # Classification
        clf = do_classification.ml_classification()
        svm_result, conf_svm, svm_model_dost = clf.do_svm(x_train_tagger_arr, x_train_tagger_arr_h, y_data,
                                                          y_data_h)  # svm
        nb_result, conf_nb, nb_model_dost = clf.do_nb(x_train_tagger_arr, x_train_tagger_arr_h, y_data,
                                                      y_data_h)  # naive bayse
        knn_result, conf_knn, knn_model_dost = clf.do_knn(x_train_tagger_arr, x_train_tagger_arr_h, y_data,
                                                          y_data_h)  # knn

    elif tokenize_type=="n":
        x_data_tagger, x_data_tagger_h, y_data_noun,y_data_noun_h = token.tokenize_(x_data, x_data_h, y_data, y_data_h, tagger,
                                                         tokenize_type)  # tokenize
        # print(x_data_tagger[:5]," x_data_tagger")
        x_train_tagger_arr, x_train_tagger_arr_h, y_train_komoran_noun, y_train_komoran_h_noun=token.tf_idf_noun(x_data_tagger, x_data_tagger_h, y_data_noun,y_data_noun_h)  # tf-idf
        # Classification
        clf = do_classification.ml_classification()
        # print(x_train_tagger_arr," x_train_komoran_noun")
        svm_result, conf_svm, svm_model_dost = clf.do_svm(x_train_tagger_arr, x_train_tagger_arr_h, y_train_komoran_noun,
                                                          y_train_komoran_h_noun)  # svm
        nb_result, conf_nb, nb_model_dost = clf.do_nb(x_train_tagger_arr, x_train_tagger_arr_h, y_train_komoran_noun,
                                                      y_train_komoran_h_noun)  # naive bayse
        knn_result, conf_knn, knn_model_dost = clf.do_knn(x_train_tagger_arr, x_train_tagger_arr_h, y_train_komoran_noun,
                                                          y_train_komoran_h_noun)  # knn

    proba_list=svm_model_dost.predict_proba(x_train_tagger_arr_h) # --> multi list using sum() to flatten
    ladol_list=svm_model_dost.predict(x_train_tagger_arr_h)
    #print(proba_list,ladol_list)

    #write to text file
    if tokenize_type=="m":
        f = open("./classification_result.txt", 'w')
        f.write("Meab with morphs"+"\n")
        f.write(svm_result+"\n")
        f.write(nb_result+"\n")
        f.write(knn_result+"\n")
        f.close()
    elif tokenize_type=="n":
        f = open("./classification_result_noun.txt", 'w')
        f.write("Meab with nouns" + "\n")
        f.write(svm_result + "\n")
        f.write(nb_result + "\n")
        f.write(knn_result + "\n")
        f.close()

    # return conf_svm, conf_nb, conf_knn, x_data_tagger_h,x_data_h
    return svm_model_dost,knn_model_dost,nb_model_dost,x_train_tagger_arr_h,svm_result,knn_result,nb_result,x_data_h

do_init("m")
