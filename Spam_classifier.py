# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 22:20:23 2020

@author: Rahul
"""
import pandas as pd
data =pd.read_csv("ML_practice\\SpamCollection",sep='\t',
                  names=['label','message'])
#data cleaning
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps=PorterStemmer()
corpus=[]
for i in range(len(data)):
    words=re.sub('[^a-zA-Z]',' ',data['message'][i])
    words=words.lower()
    words=words.split()
    words=[ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
    words=' '.join(words)
    corpus.append(words)

#creating bag of Word model
from sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer(max_features=3000)
x=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(data['label'])
y=y.iloc[:,1].values

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.20,random_state=0)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(x_train, y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)












