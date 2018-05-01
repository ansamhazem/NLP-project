# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:00:32 2018

@author: Ansam
"""
'''
e=> entertainment  
b=> bussiness
t=> science and technology 
m=> health
'''
#1. Importing all the required libraries

import re
import pandas as pd # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix,classification_report
import pickle
import predict as p
import sklearn.neighbors as knn
import plot_validation_curve as plt
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
import scikitplot as skplt
import matplotlib.pyplot as plot


#2. Function to get the words from the headlines

def get_words( headlines ):               
    headlines_onlyletters = re.sub("[^a-zA-Z]", " ",headlines) #Remove everything other than letters     
    words = headlines_onlyletters.lower().split() #Convert to lower case, split into individual words    
    stops = set(stopwords.words("english"))  #Convert the stopwords to a set for improvised performance                 
    meaningful_words = [w for w in words if not w in stops]   #Removing stopwords
    return( " ".join( meaningful_words )) #Joining the words
    
#3. Reading data and Splitting as train and test sets
def load_preprocessing(repDoc):
    news = pd.read_csv("data/uci-news-aggregator.csv") #Importing data from CSV
    news = (news.loc[news['CATEGORY'].isin(['b','e','t','m'])]) #Retaining rows that belong to categories 'b' and 'e'
    news=news[0:26000]
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(news["TITLE"], news["CATEGORY"], test_size = 0.2,random_state=53)
    X_train = np.array(X_train);
    X_test = np.array(X_test);
    Y_train = np.array(Y_train);
    Y_test = np.array(Y_test);
    cleanHeadlines_train = [] #To append processed headlines
    cleanHeadlines_test = [] #To append processed headlines
    number_reviews_train = len(X_train) #Calculating the number of reviews
    number_reviews_test = len(X_test) #Calculating the number of reviews
    #4. Getting only the words from the headlines, removing the stopwords, numbers and special characters
    
    for i in range(0,number_reviews_train):
        cleanHeadline = get_words(X_train[i]) #Processing the data and getting words with no special characters, numbers or html tags
        cleanHeadlines_train.append( cleanHeadline )
    
    for i in range(0,number_reviews_test):
        cleanHeadline = get_words(X_test[i]) #Processing the data and getting words with no special characters, numbers or html tags
        cleanHeadlines_test.append( cleanHeadline )
    #5. Creating the bag of words for each headline
    if(repDoc=='BOW'):
        vectorize = sklearn.feature_extraction.text.CountVectorizer(analyzer = "word",max_features = 1700)
        bagOfWords_train = vectorize.fit_transform(cleanHeadlines_train)
        X_train = bagOfWords_train.toarray()
        
        bagOfWords_test = vectorize.transform(cleanHeadlines_test)
        X_test = bagOfWords_test.toarray()
        pickle.dump(vectorize, open('vectorize.pickle', 'wb'))
        return X_train, X_test, Y_train, Y_test,vectorize
    elif(repDoc=='TFIDF'):
        # TFIDF
        #train
        tf_transformer = sklearn.feature_extraction.text.TfidfVectorizer(max_features=1700, strip_accents='unicode', analyzer='word',  lowercase=True, use_idf=True)
        X_train = tf_transformer.fit_transform(cleanHeadlines_train)
        pickle.dump(tf_transformer, open('tf.pickle', 'wb'))
      #test
        X_test = tf_transformer.transform(cleanHeadlines_test)
        return X_train, X_test, Y_train, Y_test,tf_transformer

    
    #------------------------------------------------------------------------
    #6. training and calculating the accuracy of the model
    #------------------------------------------------------------------------
def train(X_train, X_test, Y_train, Y_test,model,repDoc):
    if(model=='nb'):
        nb = MultinomialNB()
        nb.fit(X_train, Y_train)
        pickle.dump(nb, open('navis%s.pickle'%repDoc, 'wb'))
        print(nb.score(X_test, Y_test))
        print(nb.predict(X_test))
        print ("\n*Classification Report:\n", classification_report(Y_test, nb.predict(X_test)))

        return nb
    elif(model=='lg'):
        logistic_Regression = LogisticRegression()
        logistic_Regression.fit(X_train,Y_train)
        pickle.dump(logistic_Regression, open('logistic%s.pickle'%repDoc, 'wb'))
        Y_predict = logistic_Regression.predict(X_test)
        print("predict",Y_predict)
        print("label",Y_test)
        print(accuracy_score(Y_test,Y_predict))
        print ("\n*Classification Report:\n", classification_report(Y_test,Y_predict))
        return logistic_Regression
    elif(model=='knn'):
        n_neighbors = 11
        weights = 'uniform'
        weights = 'distance'
        KNN=knn.KNeighborsClassifier(n_neighbors, weights=weights)
        KNN.fit(X_train,Y_train)
        pickle.dump(KNN, open('%dnn%s%s.pickle'%(n_neighbors,repDoc,weights), 'wb'))
        Y_predict = KNN.predict(X_test)
        print("predict",Y_predict)
        print("label",Y_test)
        print(accuracy_score(Y_test,Y_predict))
        print ("\n*Classification Report:\n", classification_report(Y_test,Y_predict))

        return KNN
    elif(model=='svm'):
        SVM = sklearn.svm.SVC(kernel='linear', C=1, gamma=100,verbose=True) 
#        SVM = sklearn.svm.LinearSVC()
        SVM.fit(X_train,Y_train)
        pickle.dump(SVM, open('SVM%s.pickle'%repDoc, 'wb'))
        Y_predict = SVM.predict(X_test)
        print ("\n*Classification Report:\n", classification_report(Y_test, Y_predict))
        print("predict",Y_predict)
        print("label",Y_test)
        print(accuracy_score(Y_test,Y_predict))
        return SVM
    
    
def predict(sentence,model):
    Predict=model.predict(sentence)
    predict_prob= model.predict_proba(sentence)
    return Predict,predict_prob

if __name__ == '__main__':
    repDoc='TFIDF'
    model='svm'
    X_train, X_test, Y_train, Y_test ,vectorize=load_preprocessing(repDoc)
    estimator=train(X_train, X_test, Y_train, Y_test,model,repDoc)
    skplt.estimators.plot_learning_curve(estimator, X_train, Y_train)
    plot.show()
    sen=["NASA's most powerful rocket ever aims for deep space exploration"]
    print(p.predict(sen,"logisticTFIDF.pickle",repDoc))
        