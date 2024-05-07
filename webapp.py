from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained LSTM model
with open('uni_lstm.pkl', 'rb') as f:
    lstm_model = pickle.load(f)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define route for rendering index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Define route for receiving input sentences and providing suggestions
@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    input_sentence = request.json['input_sentence']

    # Use your LSTM model to predict the next word based on the input sentence
    suggestions = predict_next_word(input_sentence)

    return jsonify({'suggestions': suggestions})

def predict_next_word(input_sentence):
    # Tokenize input sentence
    token_text = tokenizer.texts_to_sequences([input_sentence])[0]
    # Truncate or pad sequences to match the model's input shape
    padded_token_text = pad_sequences([token_text], maxlen=194, padding='pre')
    # Predict
    pos = lstm_model.predict(padded_token_text)
    
    # Convert the predicted position to the actual word
    next_word = tokenizer.index_word[np.argmax(pos)]

    return next_word


if __name__ == '__main__':
    app.run(debug=True, port=3030)
