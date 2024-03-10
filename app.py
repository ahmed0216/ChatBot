from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

# Load the JSON data
with open('project_data.json') as file:
    data = json.load(file)

# Load the trained model
# model = load_model('model.h5')
model = load_model('model.h5', compile=False)    


# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as enc:
    label_encoder = pickle.load(enc)

# Function to preprocess input text using regular expressions and lemmatization
def preprocess_input(text):
    lemmatizer = WordNetLemmatizer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the sentence
    words = word_tokenize(text)
    # Lemmatize each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    # Join the lemmatized words back into a sentence
    text = ' '.join(lemmatized_words)
    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to predict intent
def predict_intent(input_text):
    input_text = preprocess_input(input_text)
    input_sequence = pad_sequences(tokenizer.texts_to_sequences([input_text]))
    result_index = np.argmax(model.predict(np.array(input_sequence), verbose=0))
    predicted_intent = label_encoder.classes_[result_index]
    return predicted_intent

# Function to get response
def get_response(predicted_intent):
    for intent in data['intents']:
        if intent['tag'] == predicted_intent:
            responses = intent['responses']
            if isinstance(responses, list):
                return np.random.choice(responses)
            else:
                return responses

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_message = request.args.get('msg')
    predicted_intent = predict_intent(user_message)
    bot_response = get_response(predicted_intent)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
