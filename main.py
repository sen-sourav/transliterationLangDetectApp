import keras.models
from flask import Flask, request, render_template
from flask import jsonify
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import string
import pyphen



# Initialize the app
app = Flask(__name__)

model = None

# load model
def load_model():
    #global variables
    global model
    jfile = open('./model/translit.json', 'r')
    loaded_model_json = jfile.read()
    jfile.close()
    model = model_from_json(loaded_model_json)
    # load weights
    model.load_weights("./model/translit.h5")
    print("Loaded model from disk")
    # compile the loaded model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def PhoneticWordSplit(word):
    dic = pyphen.Pyphen(lang='it_IT')
    splitword = dic.inserted(word)
    splitword = splitword.replace("-", " ")
    return splitword

def IntEncodeWords(wordlist):
    vocab_size = 200000
    max_length = 10
    #integer encoding the syllables
    encoded_words = [hashing_trick(d, vocab_size, hash_function='md5') for d in wordlist]
    #padding to a max length of 10
    padded_words = pad_sequences(encoded_words, maxlen=max_length, padding='post')
    return padded_words

def TextToInput(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [PhoneticWordSplit(w) for w in text]
    return IntEncodeWords(text)


@app.route('/', methods=['GET','POST'])

def predict():


    if request.method == 'POST':
        load_model()
        x_input = request.form.get('inputtext')
        if not x_input.strip():
            return render_template('index.html', inputtext=x_input, bn=0, kr=0)
            
        lstminput = TextToInput(x_input)
        langdict = {0:"Bengali", 1:"Korean"}
        p = model.predict(lstminput) 
        mle = np.log(p)
        mle = np.exp(np.sum(mle, axis=0))
        prediction = mle/np.sum(mle)

        return render_template('index.html', inputtext=x_input, bn=round(100*prediction[0],2), 
                                               kr=round(100*prediction[1],2))
                
    return render_template('index.html')



# Start the server, continuously listen to requests.

if __name__=="__main__":
    # For local development:
    app.run(debug=False)
    # For public web serving:
    #app.run(host='0.0.0.0')
