import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import streamlit as st
import pickle


file = load_iris()
x_df = pd.DataFrame(file.data,columns=file.feature_names)
y_df = pd.DataFrame(file.target,columns=['Species'])

model = pickle.load(open("model.pkl","rb"))

st.title("IRIS Flower Prediction")
st.header("please provide the input to the slider to get the class of the IRIS Flower")


def input_():
    val1 = st.sidebar.slider("{}:".format(x_df.columns[0]),min(x_df[x_df.columns[0]]),max(x_df[x_df.columns[0]]),6.64,step=0.01)

    val2 = st.sidebar.slider("{}:".format(x_df.columns[1]),min(x_df[x_df.columns[1]]),max(x_df[x_df.columns[1]]),2.63,step=0.01)

    val3 = st.sidebar.slider("{}:".format(x_df.columns[2]),min(x_df[x_df.columns[2]]),max(x_df[x_df.columns[2]]),6.43,step=0.01)

    val4 = st.sidebar.slider("{}:".format(x_df.columns[3]),min(x_df[x_df.columns[3]]),max(x_df[x_df.columns[3]]),0.5,step=0.01)

    inpt = {x_df.columns[0]:val1,
            x_df.columns[1]:val2,
            x_df.columns[2]:val3,
            x_df.columns[3]:val4}
    
    df = pd.DataFrame(inpt,index=[0])
    
    return df

df = input_()

if st.sidebar.button("SUBMIT"):
    st.write(df)
    val = model.predict(df)
    if val==0:
        st.write("Given flower is IRIS {}".format(file.target_names[0]))
    elif val==1:
        st.write("Given flower is IRIS {}".format(file.target_names[1]))
    elif val==2:
        st.write("Given flower is IRIS {}".format(file.target_names[2]))

    array = model.predict_proba(df)
    df = pd.DataFrame(array,columns=file.target_names)
    st.write(df)