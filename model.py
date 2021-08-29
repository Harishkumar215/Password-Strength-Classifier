# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 14:41:35 2021

@author: Harish
"""

import pandas as pd
import numpy as np
import random
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import f1_score
import pickle

data = pd.read_csv(r"C:\Users\Harish\Documents\Projects\Password Strength\data.csv", error_bad_lines=False)

data.head()

data['strength'].value_counts()

data.isnull().sum()

data = data.dropna(axis = 0)

passwords_tuple = np.array(data)

random.shuffle(passwords_tuple)

X = [labels[0] for labels in passwords_tuple]

y = [labels[1] for labels in passwords_tuple]


sns.set_style('whitegrid')
sns.countplot(x = 'strength', data = data, palette='RdBu_r')

def word_divide_char(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

vectorizer = TfidfVectorizer(tokenizer = word_divide_char)

X = vectorizer.fit_transform(X)

#
vectorizer.vocabulary_

first_document_vector=X[0]

feature_names = vectorizer.get_feature_names()
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])

df.sort_values(by=["tfidf"],ascending=False)

#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  #splitting

log_class = LogisticRegression(penalty='l2',multi_class='ovr')
log_class.fit(X_train,y_train)

print(log_class.score(X_test,y_test))
log_pred = log_class.predict(X_test)
f1_score(y_true= y_test, y_pred = log_pred, average = None)


clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
clf.fit(X_train, y_train) #training

print(clf.score(X_test, y_test))

log_pred_1 = clf.predict(X_test)
f1_score(y_true= y_test, y_pred = log_pred_1, average = None)


X_predict=np.array(["Harish@369"])
X_predict=vectorizer.transform(X_predict)
y_pred=log_class.predict(X_predict)
print(y_pred)



xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train,y_train)

print(xgb_model.score(X_test, y_test))

pred = xgb_model.predict(X_test)
f1_score(y_true= y_test, y_pred = pred, average = None)


# open a file, where you ant to store the data
file = open('C:/Users/Harish/Documents/Projects/Password Strength/Xgb_model1.pkl', 'wb')

# dump information to that file
pickle.dump(xgb_model, file)




























