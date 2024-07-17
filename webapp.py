import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#NLP Model
df = pd.read_csv(r'C:\Users\Elroy\Desktop\python_project\spam\spam.csv', encoding='latin1')
 

# .head() means printing first 5 rows

df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

#df.drop is removing the unnamed columns using the word function "drop"


df.rename(columns = {'v1':'labels', 'v2':'message'}, inplace = True)
#renamed the columns

#print(df.shape)
df.drop_duplicates(inplace = True)
#print(df.shape)

#seeing the number of rows and columns using "shape" and removing the duplicates

df['labels'] = df['labels'].map({'ham':0, 'spam':1})
#print(df.head())

def clean_data(message):
    message_without_punc = [character for character in message if character not in string.punctuation]
    message_without_punc = ''.join(message_without_punc)

    separator = ' '
    return separator.join([word for word in message_without_punc.split() if word.lower() not in stopwords.words('english')])


df['message'] = df['message'].apply(clean_data)

x = df['message']
y = df['labels']

cv = CountVectorizer()

x = cv.fit_transform(x)

#print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0)

model = MultinomialNB().fit(x_train, y_train)


predictions = model.predict(x_test)

#print(accuracy_score(y_test, predictions))
#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))

def predict(text):
    labels = ['Not Spam', 'Spam']
    x = cv.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str('This message is looking to be: '+labels[v])

#print(predict(['Congratulations, you won a lottery of $4000']))


st.title('Elroys Spam Checker')
st.image('image.jpg')
user_input = st.text_input('Write your message')
submit = st.button('Predict')
if submit:
    answer = predict([user_input])
    st.text(answer)












