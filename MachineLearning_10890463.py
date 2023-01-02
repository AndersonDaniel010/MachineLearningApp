# 10890463
# Anderson Daniel Torres Bernardo

#import sklearn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#We read the dataset and we set the sep and quothechar for our dataframe
df_wine = pd.read_csv('./winequality-white.csv',sep=';',quotechar='"')
print(str(df_wine.shape))
print(df_wine.head())
print("#"*100)
print()

#Doing some analysis
print(df_wine['quality'].describe())
print("#"*100)
print()

#Chech if there is any missing value
print(df_wine.isna().sum())
print("#"*100)
print()

#We need to set a range of good quality and bad quality of wine
df_wine['goodqualitywine'] = [1 if x >= 7 else 0 for x in df_wine['quality']]
print(df_wine.head(20))
print("#"*100)
print()

#We can even check how many good qualitywines do we have
print(df_wine['goodqualitywine'].value_counts())
print("#"*100)
print()

#Set our attributes and target
attribute = df_wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
target = df_wine['goodqualitywine']

#Spliting the data into TEST and TRAIN
X_train, X_test, y_train, y_test = train_test_split(attribute, target, test_size=.25, random_state=0)

#Model is with randomForest
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

#Prediction report F1 score included
print(classification_report(y_test, y_pred2))
print("#"*100)
print()

########################################################################
################# Machine Learing APP made in sreamlit #################
########################################################################
st.title('My Machine Learning App using streamlit')

fixed_acidity = st.number_input(
    "Fixed acidity",
    min_value=0.0,
    step=1e-6,
    format="%.5f",
    key=1)

volatile_acidity = st.number_input(
    "Volatile acidity",
    min_value=0.0,
    step=1e-6,
    format="%.5f",
    key=2) 

citric_Acid = st.number_input(
    "Citric acidity",
    min_value=0.0,
    step=1e-6,
    format="%.5f",
    key=3)

residual_sugar = st.number_input(
    "Residual Sugar",
    min_value=0.0,
    step=1e-6,
    format="%.5f",
    key=4)

chlorides = st.number_input(
    "Chlorides",
    min_value=0.0,
    step=1e-6,
    format="%.5f",
    key=5)

totalsulfurdioxide = st.number_input(
    "Total Sulfur Dioxide",
    min_value=0.0,
    step=1e-6,
    format="%.5f",
    key=6)

freesulfurdioxide = st.number_input(
    "Free Sulfur Dioxide",
    min_value=0.0,
    step=1e-6,
    format="%.5f",
    key=7)

density = st.number_input(
    "Density",
    min_value=0.0,
    step=1e-6,
    format="%.5f",
    key=8)

pH = st.number_input(
    "pH",
    min_value=0.0,
    step=1e-6,
    format="%.5f",
    key=9)


sulphates = st.number_input(
    "Sulphates",
    min_value=0.0,
    step=1e-6,
    format="%.5f",
    key=10)

alcohol = st.number_input(
    "Alcohol",
    min_value=0.0,
    step=1e-6,
    format="%.5f",
    key=11)

df_wine_new = df_wine[['fixed acidity', 'volatile acidity', 'citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]

columns=list(df_wine_new.columns)
print(columns)

def predict():
    row=np.array([fixed_acidity, volatile_acidity, citric_Acid, residual_sugar, chlorides, freesulfurdioxide,totalsulfurdioxide, density, pH, sulphates, alcohol])
    X = pd.DataFrame([row], columns=columns)
    prediction = model2.predict(X)[0]

    if prediction == 1:
        st.success('Good Quality Wine')
        st.markdown('Hi')
        st.ballons()
        
    else:
        st.error('Not good quality Wine')
        st.markdown('Hi')
        
st.button('Make Prediction', on_click=predict)




