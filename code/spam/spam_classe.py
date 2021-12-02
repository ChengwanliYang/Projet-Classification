#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import stop_words
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn import feature_extraction
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#1. lire le fichier

data_spam =pd.read_csv("spamham.csv")
y = data_spam["spam"]

#2. Vectoriser en unigramme, stopword true, lowercase false.
V1 = CountVectorizer(stop_words="english")
V2 = TfidfVectorizer(stop_words="english")

#3. separer train et test. 

liste_classifieurs= [
    ["Perceptron", Perceptron(eta0=0.1, random_state=0)],
    ["naive_bayes", MultinomialNB()],    
    ["Logistic Regression", LogisticRegression()]
]
for V in (V1, V2):
        X = V.fit_transform(data_spam["text"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
#4. mesuser 
        for nom, algo in liste_classifieurs:
            clf = algo.fit(X_train, y_train)
            y_pred = algo.predict(X_test)
            
            matrice_confusion = confusion_matrix(y_test, y_pred)
            print(matrice_confusion)
            
            report = classification_report(y_test, y_pred)
            nom_classes = ["ham", "spam"]
            report = classification_report(y_test, y_pred, target_names=nom_classes)
            print(report)
 


# In[ ]:




