#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import feature_extraction
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

#1. lire le fichier

df_chunk=pd.read_csv("french_tweets.csv",chunksize=10000)
res_chunk=[]
for chunk in df_chunk:
    res_chunk.append(chunk)
    data_tweet=pd.concat(res_chunk)
y = data_tweet["label"]

#2. Vectoriser
V1 = CountVectorizer(ngram_range=(1, 3),lowercase=True)
V2 = TfidfVectorizer(ngram_range=(1, 3),lowercase=True)
warnings.filterwarnings("ignore")

#3. separer en unigramme,bigramme et trigramme, stopword false, lowercase true 

liste_classifieurs= [
    
    ["Perceptron", Perceptron(eta0=0.1, random_state=0)],
    ["naive_bayes", MultinomialNB()],    
    ["Logistic Regression", LogisticRegression()]   
]

for V in (V1, V2):
        X = V.fit_transform(data_tweet["text"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
#4. mesuser 
        for nom, algo in liste_classifieurs:
            clf = algo.fit(X_train, y_train)
            y_pred = algo.predict(X_test)
            
            matrice_confusion = confusion_matrix(y_test, y_pred)
            print(matrice_confusion)
            
            report = classification_report(y_test, y_pred)
            nom_classes = ["negatif", "positif"]
            report = classification_report(y_test, y_pred, target_names=nom_classes)
            print(report)
 
       


# In[ ]:




