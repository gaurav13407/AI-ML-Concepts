import streamlit as st
import pandas as pd
st.title("Streamlit Text Input")

name=st.text_input("Enter your name:")

age=st.slider("select your age",0,100,25)
st.write(f"your age is{age}")

options=["python","java","c++","javascript"]
choice=st.selectbox("Choose your favorite launguage:",options)
st.write(f"You selected {choice}")
if name:
    st.write(f"Hello,{name}")
    
data={
    "Name":["john","jane","jake","Jill"],
    "age":[28,24,35,40],
    "City":["New Yourk","los angles","chicago","Houston"]
}

df=pd.DataFrame(data)
df.to_csv("sampledata.csv")
st.write(df)

uploaded_file=st.file_uploader("Choose a csv file",type="csv")

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df)