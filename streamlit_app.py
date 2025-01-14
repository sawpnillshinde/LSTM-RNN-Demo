import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding,LSTM,Dropout,Dense 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import streamlit as st

model=load_model('next_word_lstm.h5')


with open ('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

def predict_next_word(model,tokenizer,text,max_sequence_length):
    token_list=tokenizer.texts_to_sequences([text])[0]
    #max_sequence_length=max(len(word) for word in token_list)
    if len(token_list) >= max_sequence_length:
        token_list=token_list[-(max_sequence_length-1):]
    token_list=pad_sequences([token_list],maxlen=max_sequence_length-1,padding='pre')
    prediction=model.predict(token_list,verbose=1)
    predicted_word_index=np.argmax(prediction,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    else:
        return None
    
    #Streamlit App
st.title("Next Word Prediction App")
input_text=st.text_input("Enter the sequence of words/sentence for prediction","Eg:TO be or not to be")
if st.button('Predict'):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'Next Word is : {next_word}')