import os
from flask import jsonify
from flask import request
from flask import Flask
import numpy as np

from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

import pickle

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

np.random.seed(1337)

graph = tf.get_default_graph()


#star Flask application
app = Flask(__name__)

#Load model
path = 'G:/Script/LeadIQ'
json_file = open(path+'/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
keras_model_loaded = model_from_json(loaded_model_json)
keras_model_loaded.load_weights(path+'/model.h5')
print('Model loaded...')


#load tokenizer pickle file
with open(path+'/tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)

tok.oov_token = None
def preprocess_text(texts,max_review_length = 100):
    cnn_texts_seq = tok.texts_to_sequences(texts)
    cnn_texts_mat = pad_sequences(cnn_texts_seq,maxlen=max_review_length)
    return cnn_texts_mat

# URL that we'll use to make predictions using get and post
#@app.route('/predict',methods=['GET','POST'])
@app.route('/predict',methods=['POST'])

def predict():
    text = request.args.get('text')
    x = preprocess_text([text])
    with graph.as_default():
        return jsonify({'prediction': str(keras_model_loaded.predict(x)[0])})


if __name__ == "__main__":
    # Run locally
    app.run(debug=False)

