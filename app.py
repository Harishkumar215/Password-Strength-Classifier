# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 16:31:13 2021

@author: Harish
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import xgboost as xgb


data = pd.read_csv(r"C:\Users\Harish\Documents\Projects\Password Strength\data.csv", error_bad_lines=False)
data = data.dropna(axis = 0)

passwords_tuple = np.array(data)

random.shuffle(passwords_tuple)

X = [labels[0] for labels in passwords_tuple]

y = [labels[1] for labels in passwords_tuple]

def word_divide_char(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

vectorizer = TfidfVectorizer(tokenizer = word_divide_char)

X = vectorizer.fit_transform(X)


st.write("# Password Strength classifier")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  #splitting

xgb_model = xgb.XGBClassifier()

xgb_model.fit(X_train,y_train)


password = st.text_input("Enter the Password", value='Abcd@123')

st.write(password)

X_predict = np.array([password])

X_predict = vectorizer.transform(X_predict)

y_pred = xgb_model.predict(X_predict)

st.write("""# Password Class
         * 0 - Weak
         * 1 - Medium
         * 2 - Strong""")
         
st.write(y_pred)
