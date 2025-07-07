import pandas as pd
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english')]
    return " ".join(y)

# Load dataset
df = pd.read_csv("spam (1).csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['transformed_text'] = df['text'].apply(transform_text)

# Train TF-IDF and Model
X = df['transformed_text']
y = df['label']

tfidf = TfidfVectorizer()
X_vectorized = tfidf.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

# Save vectorizer and model correctly
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(" Model and Vectorizer saved successfully.")
