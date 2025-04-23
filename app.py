# @Abhishek pp
# github-abiek12


from flask import Flask, render_template, request
import pandas as pd
import sklearn
import itertools
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load the model and vectorizer using the correct paths
model_path = r"C:\Users\srikr\Downloads\Fake-News-Detection-using-MachineLearning-main\Fake-News-Detection-using-MachineLearning-main\model.pkl"
vector_path = r"C:\Users\srikr\Downloads\Fake-News-Detection-using-MachineLearning-main\Fake-News-Detection-using-MachineLearning-main\vector.pkl"

# Load the model and vectorizer using pickle
loaded_model = pickle.load(open(model_path, 'rb'))
vector = pickle.load(open(vector_path, 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

def fake_news_det(news):
    try:
        # Preprocess the input text
        review = re.sub(r'[^a-zA-Z\s]', '', news)  # Remove non-alphabetic characters
        review = review.lower()  # Convert to lowercase
        review = nltk.word_tokenize(review)  # Tokenize the text
        corpus = [lemmatizer.lemmatize(word) for word in review if word not in stpwrds]  # Lemmatize and remove stopwords
        
        # Debugging line to print the processed input
        print(f"Processed Input: {' '.join(corpus)}")  # This will print the tokenized and lemmatized input
        
        # Check if corpus is empty after processing
        if not corpus:
            print("Processed input is empty after preprocessing.")
            return [0]  # Return real news if empty
        
        # Vectorize the input data
        input_data = [' '.join(corpus)]
        vectorized_input_data = vector.transform(input_data)  # Transform using the saved vectorizer
        
        # Predict the class (fake or real news)
        prediction = loaded_model.predict(vectorized_input_data)
        return prediction
    except Exception as e:
        print(f"Error: {e}")  # If an error occurs, print it
        return [0]  # Default to real news if there's an error


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        print(f"Input News: {message}")  # Debug: show the input news
        pred = fake_news_det(message)  # Get the prediction
        print(f"Model Prediction: {pred}")  # Debug: show raw prediction
        
        # Check and interpret the prediction
        if pred[0] == 1:
            result = "Prediction of the News: Looking Fake News 📰"
        else:
            result = "Prediction of the News: Looking Real News 📰"
        
        # Render the result to the prediction page
        return render_template("prediction.html", prediction_text=result)
    else:
        return render_template('prediction.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
