# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:47:55 2018

@author: Ansam
"""
import pickle

def predict(sen,pickfile,repDoc):
    if(repDoc=='BOW'):
        loaded_vectorize = pickle.load(open('vectorize.pickle', 'rb'))
        vector = loaded_vectorize.transform(sen)
        vb = vector.toarray()
        loaded_model = pickle.load(open(pickfile, 'rb'))
        Y_predict=loaded_model.predict(vb)
        print(Y_predict)
    elif(repDoc=='TFIDF'):
        tf_transformer = pickle.load(open('tf.pickle', 'rb'))
        X_sen = tf_transformer.transform(sen)
        loaded_model = pickle.load(open(pickfile, 'rb'))
        Y_predict=loaded_model.predict(X_sen)
        print(Y_predict)
    return Y_predict

def predict_prob(sen,pickfile,repDoc,model):
    if(repDoc=='BOW'):
        loaded_vectorize = pickle.load(open('vectorize.pickle', 'rb'))
        vector = loaded_vectorize.transform(sen)
        vb = vector.toarray()
        loaded_model = pickle.load(open(pickfile, 'rb'))
        Y_predict=loaded_model.predict_proba(vb)
        print(Y_predict)
    elif(repDoc=='TFIDF'):
        tf_transformer = pickle.load(open('tf.pickle', 'rb'))
        X_sen = tf_transformer.transform(sen)
        loaded_model = pickle.load(open(pickfile, 'rb'))
        Y_predict=loaded_model.predict_proba(X_sen)
        print(Y_predict)
    return Y_predict