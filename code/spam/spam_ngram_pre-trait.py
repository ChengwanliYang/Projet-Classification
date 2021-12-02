#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import warning
data_spam =pd.read_csv("spamham.csv") #lire le fichier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
y = data_spam["spam"]
#print(y)  #instence et classe

from sklearn.feature_extraction.text import CountVectorizer
V = CountVectorizer()
X = V.fit_transform(data_spam["text"])
#print(X.shape)  #instence et mot

## séparer train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#print(X_train.shape[0], X_test.shape[0])  #taille de train et test 


# In[2]:


#on choisit 3 classifieurs

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
liste_classifieurs= [
    ["Perceptron", Perceptron(eta0=0.1, random_state=0)],
    ["naive_bayes", MultinomialNB()],    
    ["Logistic Regression", LogisticRegression()]
]


# In[7]:


# tester n-gramme

en_minuscules,enlever_stopwords  = False, False

for min_N in range(1, 2):
  for max_N in range(min_N, 4):
    V = CountVectorizer(ngram_range = (min_N, max_N))
    X = V.fit_transform(data_spam["text"])
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


# l'unigramme est le meilleur.

# In[8]:


#comparer les accuracys des 3 classifieurs en unigramme.


from sklearn.feature_extraction import stop_words

min_N, max_N = 1, 1

for enlever_stopwords in [False, True]:
  liste_stopwords = None
  if enlever_stopwords==True:
    liste_stopwords = stop_words.ENGLISH_STOP_WORDS
    
  for en_minuscules in [False, True]:
    print(f"Stopwords {enlever_stopwords}, Minuscules : {en_minuscules}")
    warnings.filterwarnings("ignore")
    for max_F in [100]:
        V = CountVectorizer(lowercase = en_minuscules, stop_words =  liste_stopwords, max_features = max_F )
        X = V.fit_transform(data_spam["text"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
        for nom, algo in liste_classifieurs:
            clf = algo.fit(X_train, y_train)
            expe = str([nom, min_N, max_N, enlever_stopwords, en_minuscules, max_F])
            print('  %s classifier : %.4f'%(nom, score))
            score = clf.score(X_test, y_test)
print("-"*20)



# 
