import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import kaggle

# df = pd.read_csv('./data/creditcard.csv')

# Add this to your code to download the dataset from Kaggle
import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Download dataset if it doesn't exist
if not os.path.exists('data/creditcard.csv'):
    # Set up Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download the dataset
    api.dataset_download_files(
        'mlg-ulb/creditcardfraud', 
        path='data', 
        unzip=True
    )
    print("Dataset downloaded from Kaggle")

# Load the dataset
df = pd.read_csv("data/creditcard.csv")


# separating the data for analysis
legit = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

# separating the data for training and testing
legit_sample = legit.sample(n=492)
new_df = pd.concat([legit_sample, fraud], axis=0)

# dividing the data into features and labels
X = new_df.drop(columns='Class', axis=1)
y = new_df['Class']

# train test split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# st.write(f"Train Accuracy: {train_acc}")
# st.write(f"Test Accuracy: {test_acc}")

# Create streamlit app 
st.title("Credit Card Fraud Detection")
st.write("Enter the details of the transaction to check if it is a fraud or not:")

# Create input fields for user enter features values
input_df = st.text_input('Input all features:')
input_df_lst = input_df.split(',')

# create a button to submit input and get prediction 
submit = st.button('Submit')

if submit:
    np_df = np.asarray(input_df_lst, dtype=np.float64)
    prediction = model.predict(np_df.reshape(1, -1))

    if prediction[0] == 0:
        st.write("The transaction is not a fraud.")
    else:
        st.write("The transaction is a fraud.")
    