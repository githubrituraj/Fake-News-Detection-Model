from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the model and vectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
def stemming(content):
    if not isinstance(content, str):
        return ""
    # Remove non-alphabetic characters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

model = joblib.load("lr_model.jb")
vectorizer = joblib.load("vectorizer_new.jb")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news_input']

    news_stripped = news.strip()
    if news_stripped == "":
        result = "Please enter some text to analyse."
    elif len(news_stripped.split()) < 8:
        result = "Input is too short for testing. Please provide a longer news article."
    else:
        cleaned = stemming(news)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)
        result = "Real News " if prediction[0] == 1 else "Fake News "

    return render_template('index.html', prediction=result, news=news)

if __name__ == '__main__':
    app.run(debug=True)

