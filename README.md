# News Article Classifier

This project is a web application for classifying news articles into categories such as World, Sports, Business, and Sci/Tech using machine learning models. The application is built using Python, Flask, HTML, CSS, and Scikit-Learn.

## Project Structure

```
news-article-classifier/
│
├── app.py                    # Flask application
├── templates/
│   ├── index.html            # HTML template for the home page
│   ├── result.html           # HTML template for the results page
│
├── static/
│   ├── styles.css            # CSS file for styling
│
├── models/
│   ├── logistic_regression_model.pkl  # Trained Logistic Regression model
│   ├── naive_bayes_model.pkl          # Trained Naive Bayes model
│   ├── svm_model.pkl                  # Trained SVM model
│   ├── tfidf_vectorizer.pkl           # Trained TF-IDF vectorizer
│
├── train.csv                 # Training dataset
├── test.csv                  # Testing dataset
├── evaluate_models.py        # Script to train and evaluate models
├── README.md                 # This README file
```

## Setup Instructions

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/news-article-classifier.git
    cd news-article-classifier
    ```

2. **Install the required libraries:**

    Ensure you have Python 3.6+ and install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download NLTK stopwords:**

    ```python
    import nltk
    nltk.download('stopwords')
    ```

4. **Train the models (optional):**

    If you want to retrain the models, run the `evaluate_models.py` script:

    ```bash
    python evaluate_models.py
    ```

5. **Run the Flask application:**

    ```bash
    python app.py
    ```

6. **Open the application in your browser:**

    Navigate to `http://127.0.0.1:5000/` in your web browser.

## Usage

1. **Home Page:**

    - Enter a news article text in the textarea.
    - Click the "Predict" button.

2. **Results Page:**

    - View the predicted category from Logistic Regression, Naive Bayes, and SVM models.
    - Click "Back" to return to the home page.

## Models Used

- **Logistic Regression**
- **Naive Bayes**
- **Support Vector Machine (SVM)**

These models are trained using the `train.csv` dataset and evaluated using the `test.csv` dataset. The text data is preprocessed and vectorized using TF-IDF.

## Preprocessing Steps

1. Remove HTML tags.
2. Remove non-alphabetic characters.
3. Convert text to lowercase.
4. Remove stopwords.
5. Vectorize text using TF-IDF.

## Evaluation Metrics

The models are evaluated using accuracy, precision, recall, and F1-score. The confusion matrix is also provided for each model.

## Visualization

The accuracy of the models is visualized using a bar chart. You can generate this visualization by running the `evaluate_models.py` script.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- The dataset used in this project is the AG News dataset, available on [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset).
- This project uses the following libraries: Flask, Scikit-Learn, NLTK, Pandas, Matplotlib.
