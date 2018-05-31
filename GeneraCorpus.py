# coding: utf8

__author__="Miguel Sánchez"

import re,sys, os
from sklearn.externals import joblib
import codecs
import numpy as numpy
from builtins import str
from collections import defaultdict
from sklearn import svm
from Preprocesamiento.tokenizer import tokenizar
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from Preprocesamiento.lematizaNoticias import lematizar
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import chi2
#from sklearn.model.selection import kfold
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
data_train_original = []
data_test_original = []
clases = ["cartera","ciencias","cultura","destinos","deultima","estados","techbit","deportes","espectaculos","menu"]
#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
cols = ['ID','sección', 'noticia','']
"""a=open("sincero.txt",'x')
r=open("/home/brendamv/Descargas/SP_Ranks.txt",'r')
a.write(lematizar(tokenizar(r.read())))
r.close()
a.close()"""
df = pd.read_csv('/home/brendamv/Descargas/la-jornada.txt', sep='łłłłł', header=None, engine='python',names=cols) 
feature_vector = CountVectorizer()
clf=svm.LinearSVC(dual=False,random_state=1,C=1000) 
clf.fit(feature_vector.fit_transform(df['noticia']),df['sección'])
joblib.dump(clf,"ClasificadorJornada.pkl")
clf=joblib.load("ClasificadorJornada.pkl")
print(clf)
#Separa el corpus cargado en el DataFrame en el 90% para entrenamiento y el 10% para pruebas
#print(df)
"""data_train, data_test, y_train, y_test = \
train_test_split( df['noticia'],df['sección'], test_size=0.1, random_state=0)"""

kf = StratifiedKFold(n_splits=10)
acc=0
acc1=0
acc2=0
acc3=0

for train_index, test_index in kf.split(df['noticia'], df['sección']):
    x_train, y_train = df['noticia'][train_index], df['sección'][train_index] 
    x_test, y_test = df['noticia'][test_index], df['sección'][test_index]
#Entrenamiento y prueba

#feature_vector = TfidfVectorizer()
#print(data_train)
#~ feature_vector = CountVectorizer(binary = True)

#sel = VarianceThreshold(threshold=(.95* (1 - .95)))
#sel=SelectKBest(chi2, k=17000)
#X_train = feature_vector.fit_transform(data_train).toarray()
#X_train2=sel.fit_transform(X_train,y_train)
    X_train = feature_vector.fit_transform(x_train).toarray()
#X_train=X_train2
    X_test = feature_vector.transform(x_test).toarray()
#X_test=sel.transform(X_test)
#print(X_test)
#multi_class='crammer_singer'

#clasificador es el que ustedes definan
#clf = clasificador.fit(X_train, corpus.class_train, corpus.classes)
#print(predictions)
#clasification_report(clases reales,predichas,clases)
#sklearn.metrics clasification_report
#precision y recall

#CUARTA ITERACION---------------------------------------------
    clf=svm.LinearSVC(dual=False,random_state=1,C=1000)
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    acc+=accuracy
    print ("SVM LINEAL : %f"%accuracy)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test,predictions))

#QUINTA ITERACION--------------------------------------------------------------------------
    clf=linear_model.LogisticRegression(C=15000)
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print ("REGRESION LOGISTICA : %f"%accuracy)
    print(classification_report(y_test, predictions))
    acc1+=accuracy
#print(confusion_matrix(y_test,predictions))
#SEXTA ITERACION---------------------------------------------------------------------------
    clf=MultinomialNB()
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print ("MULTINOMIAL BAYES :  %f"%accuracy)
    print(classification_report(y_test, predictions))
    acc2+=accuracy
#print(confusion_matrix(y_test,predictions,labels=y_test))
    clf1=MultinomialNB()
    clf2=linear_model.LogisticRegression(C=15000)
    clf3=svm.LinearSVC(dual=False,random_state=1,C=1000)

    eclf1 = VotingClassifier(estimators=[
         ('MNB', clf1),('LR', clf2), ('SVM', clf3)],n_jobs=-1 )
    eclf1 = eclf1.fit(X_train, y_train)
    predictions=eclf1.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print ("ESSEMBLE ACCURACY :  %f"%accuracy)
    acc3+=accuracy
print("ACCURACY PROMEDIO SVM LINEAL %f"%(acc/10.0))
print("ACCURACY PROMEDIO REGRESIÓN LOGÍSTICA %f"%(acc1/10.0))
print("ACCURACY PROMEDIO NAÏVE BAYES %f"%(acc2/10.0))
print("ACCURACY PROMEDIO ESSEMBLE %f"%(acc3/10.0))
