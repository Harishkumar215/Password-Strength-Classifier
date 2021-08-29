# Password-Strength-Classifier
Password Strength Classifier using XGBoost and Streamlit

* Created a tool that calssify strength of the password
*  Optimized Logistic, Gradient Boosting classifier using GirdSearchCV to reach the best model
*  Built a client app using Streamlit


# Code and Resources used
Pythoon Version: 3.8 \
Packages: pandas, numpy, sklearn, XGB, matplotlib, seaborn,  pickle

# Data
Downloaded data from www.kaggle.com/bhavikbb/password-strength-classifier-dataset

Column names:

* password
* strength

They are three different classess:

0 - Weak, 1 - Medium, 2 - Strong

# Data Cleaning

* Removed null values
* Converted string into list of characters and vectorized using TfidfVectorizer

# EDA
I looked at the distribution of data and below are the few highlights: \
![Alt Text](https://github.com/Harishkumar215/Password-Strength-Classifier/blob/main/Figure%202021-08-29%20154533.png)

# Model Building
From EDA I found out that the dataset is unbalanced, I used K fold cross valation to split the train and test set and trained the model

I tired different models and evaluated them using F1 score

I tried following models:

* Logistic Classifier - Baseline for the model
* XGBoost

# Model Performance
XGBoost performed better on the test and validation sets

* Logistic Classifier F1 score - [0.39872153, 0.88493897, 0.74741099]
* XGBoost - [0.96384275, 0.99101807, 0.98532687]

# Production
Built a Streamlit API endpoint that was hosted on the local web server, API endpoint takes in password and returns password strength

