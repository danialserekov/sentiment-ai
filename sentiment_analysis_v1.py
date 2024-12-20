import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download

# Load some the necessary resources for nltk
download('punkt') # for tokens to divide text for separate token
download('stopwords') # most recent words like ("and", "in", "on")

# Uploading csv dataset file with 500 thousand hundred line of reviews
data = pd.read_csv('Reviews.csv') # Amazon reviews Dataset 568,454 users

# Limit it for better capacity and fast loading
data = data.head(10000) # first 10 thousand lines

# Dividing Sentiment for Positive / Neutral / Negative
def label_sentiment(score):
    if score >= 4:
        return 'Positive' # Score must be more or equal from 4
    elif score == 3:
        return 'Neutral' # Score must be equal to 3
    else:
        return 'Negative' # Score must be less than 3

data['Sentiment'] = data['Score'].apply(label_sentiment) # converts sentiments

# Tokenization process of text
def preprocess_text(text):
    tokens = word_tokenize(text.lower()) # converts text to lower case
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word.isalnum() and word not in stop_words])

# Apply tokenization for text
data['ProcessedText'] = data['Text'].apply(preprocess_text)

# Split the data
X = data['ProcessedText']
y = data['Sentiment']

# Convert text to numeric signs
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Training models with balanced classes
lr_model = LogisticRegression(max_iter=10000, class_weight='balanced')
svm_model = SVC(kernel='linear', class_weight='balanced')
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

# Training each model
lr_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)

# Model Evaluation
y_pred_lr = lr_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_nn = nn_model.predict(X_test)

# Output results of each model using zero_division=0 to handle uncertain accuracy
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr, zero_division=0))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm, zero_division=0))
print("Neural Network Classification Report:\n", classification_report(y_test, y_pred_nn, zero_division=0))

# Samples to check sentiment of reviews
sample_reviews = [
    "This product is great! I love it.",
    "It was okay, not amazing, but decent.",
    "Worst purchase ever. Totally disappointed.",
    "Really good, but not perfect. I would recommend it.",
    "Absolutely terrible, I will never buy it again.",
    "Wtf is it",
    "Wow, beast",
    "Normal, clean",
    "Damn it, best"
]

# Sentiment Prediction
sample_reviews_processed = [preprocess_text(review) for review in sample_reviews]
sample_reviews_tfidf = vectorizer.transform(sample_reviews_processed)

# Prediction for logistic regression
sentiments_lr = lr_model.predict(sample_reviews_tfidf)
print("\nPredicted Sentiments (Logistic Regression):", sentiments_lr)

# Prediction for support vector machine
sentiments_svm = svm_model.predict(sample_reviews_tfidf)
print("\nPredicted Sentiments (SVM):", sentiments_svm)

# Prediction for neural network
sentiments_nn = nn_model.predict(sample_reviews_tfidf)
print("\nPredicted Sentiments (Neural Network):", sentiments_nn)
