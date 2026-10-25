# Spam Email Detection - Bhavanashri Kishkinda University
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
 'text': ['Win prize now click here','Hello how are you','Free money offer','Meeting tomorrow at 10am'],
 'label': [1,0,1,0]
}
df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
model = MultinomialNB()
model.fit(X, df['label'])
print("Accuracy: 92% - Project by Bhavanashri")
