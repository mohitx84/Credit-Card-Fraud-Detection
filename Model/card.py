import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# reading data
credit_card= pd.read_csv('creditcard.csv/creditcard.csv')
# legit and fraud analysis
legit= credit_card[credit_card.Class==0]
fraud= credit_card[credit_card.Class==1]
# undersampling legit transaction to make it good
legit_new = legit.sample(n=len(fraud),random_state=0)
credit_card = pd.concat([legit_new,fraud],axis=0)

# split data into training and testing states
X= credit_card.drop('Class',axis=1)
y= credit_card['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic model
model= LogisticRegression()
model.fit(X_train,y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train),y_train)
test_acc = accuracy_score(model.predict(X_test),y_test)

# create Streamlit app
st.title("Credit card fraud detection")
input_df= st.text_input("Enter the transaction details")
input_df_splited= input_df.split(",")

submit= st.button("Submit")
if submit:
 details= (np.asarray(input_df_splited,dtype=np.float64))
 prediction = model.predict(details.reshape(1, -1))
 if prediction[0] ==0:
        st.write("This is a legit transaction")
 elif prediction[0] ==1:
    st.write("This is a fraud transaction!!!!")
 else:
     st.write("Enter valid transaction details")



