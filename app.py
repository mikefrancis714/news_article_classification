import re
from nltk.corpus import stopwords
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the models and vectorizer
lr_model = joblib.load('models/logistic_regression_model.pkl')
nb_model = joblib.load('models/naive_bayes_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')


# Class labels
class_labels = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


# Preprocess the data
def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        cleaned_message = preprocess_text(message)
        vect_message = vectorizer.transform([cleaned_message])

    
    # Prediction from each model
    lr_prediction = lr_model.predict(vect_message)[0]
    nb_prediction = nb_model.predict(vect_message)[0]
    svm_prediction = svm_model.predict(vect_message)[0]

    # Map prediction to class labels
    lr_prediction_label = class_labels[lr_prediction]
    nb_prediction_label = class_labels[nb_prediction]
    svm_prediction_label = class_labels[svm_prediction]
    
    return render_template('result.html',
                           lr_prediction=lr_prediction_label,
                           nb_prediction=nb_prediction_label,
                           svm_prediction=svm_prediction_label,)

if __name__ == "__main__":
    app.run(debug=True)
