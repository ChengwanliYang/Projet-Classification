#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import feature_extraction


df_chunk=pd.read_csv("french_tweets.csv",chunksize=10000)
res_chunk=[]
for chunk in df_chunk:
    res_chunk.append(chunk)
    data_tweet=pd.concat(res_chunk)
y = data_tweet["label"]    

liste_classifieurs= [
    ["Perceptron", Perceptron(eta0=0.1, random_state=0)],
    ["Logistic Regression", LogisticRegression()],
    ["naive_bayes", MultinomialNB()] 
]
en_minuscules,enlever_stopwords  = False, False
for min_N in range(1, 2):
  for max_N in range(min_N, 4):
    V = CountVectorizer(ngram_range = (min_N, max_N))
    X = V.fit_transform(data_tweet["text"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    print( "-" * 15)
    print(f"Ngram_range : ({min_N}, {max_N})")
    warnings.filterwarnings("ignore")
    for nom, algo in liste_classifieurs:
        expe = str([nom, min_N, max_N, enlever_stopwords, en_minuscules])
        clf = algo.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('  %s classifier : %.4f'%(nom, score))
        pred = clf.predict(X_test)



# In[4]:


print(0.7118+0.7871+0.7692)
print(0.7587+0.8025+0.7853)
print(0.7771+0.8023+0.7883)


# Utiliser l’unigramme, le bigramme et le trigramme.

# In[6]:


min_N, max_N = 1, 3

for enlever_stopwords in [False, True]:
  liste_stopwords = None
  if enlever_stopwords==True:
    liste_stopwords = stopwords.words('french')
    
  for en_minuscules in [False, True]:
    print(f"Stopwords {enlever_stopwords}, Minuscules : {en_minuscules}")
    for max_F in [100]:
        V = CountVectorizer(lowercase = en_minuscules, stop_words =  liste_stopwords, max_features = max_F )
        X = V.fit_transform(data_tweet["text"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
        for nom, algo in liste_classifieurs:
            clf = algo.fit(X_train, y_train)
            expe = str([nom, min_N, max_N, enlever_stopwords, en_minuscules, max_F])
            print('  %s classifier : %.4f'%(nom, score))
            score = clf.score(X_test, y_test)
                
print("-"*20)



# In[7]:


print(0.6465+0.5391+0.6462)
print(0.6313+0.5429+0.6588)
print(0.6465+0.5551+0.6222)
print(0.5926+0.5561+0.6335)


# le meilleur cas : Stopwords False, Minuscules : True

# In[ ]:




