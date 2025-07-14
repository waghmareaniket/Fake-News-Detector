import numpy as np
import pandas as pd 
import re
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv('train.csv')
data=data.fillna(' ')
data['content']=data['author']+" "+data['title']
x=data.iloc[:,:-1]
y=data['label']

#stem
ps=PorterStemmer()
def stemming(content):
    stem_content=re.sub('[^a-zA-Z]'," ",content)
    stem_content=stem_content.lower()
    stem_content=stem_content.split()
    stem_content=[ps.stem(word)for word in stem_content if not word in stopwords.words('english')]
    stem_content=" ".join(stem_content)
    return stem_content


data['content']=data['content'].apply(stemming)

x=data['content'].values
y=data['label'].values
vector=TfidfVectorizer()
vector.fit(x)
x=vector.transform(x)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y, random_state=1)

model=LogisticRegression()
model.fit(X_train,y_train)

# website
st.title('Fake News Detector')
input=st.text_input('Please Enter News Article')
def predictor(input):
    input=vector.transform([input])
    predictor=model.predict(input)
    return predictor[0]

if input:
    pred=predictor(input)
    if pred==1:
        st.warning('fake news') 
    else:
        st.success( 'Real news')


# train_y_pred=model.predict(X_train)
# print('training data: 'accuracy_score(train_y_pred,y_train))

# test_y_pred=model.predict(X_test)
# print('training data: 'accuracy_score(test_y_pred,y_test))
