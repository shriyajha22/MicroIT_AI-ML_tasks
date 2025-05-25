import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords

# Sample dataset
data = {
    'text': [
        "I love this product!", 
        "This is the worst thing I ever bought.", 
        "Absolutely fantastic service.", 
        "Not great, not terrible.", 
        "I am really disappointed.", 
        "Best experience ever!", 
        "It's okay, nothing special.",
        "I'm not happy with this.",
        "Superb quality and fast delivery.",
        "Terrible. I want a refund."
    ],
    'sentiment': [
        "positive", "negative", "positive", "neutral", "negative",
        "positive", "neutral", "negative", "positive", "negative"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Text Preprocessing
def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Real-time Prediction
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

# Example
while True:
    user_input = input("\nEnter a sentence for sentiment analysis (or 'exit'): ")
    if user_input.lower() == 'exit':
        break
    sentiment = predict_sentiment(user_input)
    print(f"Predicted Sentiment: {sentiment}")
