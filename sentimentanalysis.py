import pickle
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyze = SentimentIntensityAnalyzer()

from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random

from keras.models import model_from_json

model = load_model('chatbot_model.h5')

model_json = model.to_json()
with open('chatbot_model.json', 'w') as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights('chatbot_model.h5')

print("Model and weights saved")


from keras.models import model_from_json

# Load the model architecture from JSON
with open('chatbot_model.json', 'r') as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json)

# Load the model weights
model.load_weights('chatbot_model_weights.h5')

print("Model and weights loaded")



nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('/kaggle/input/chatbot-dataset/intents.json').read())

words = pickle.load(open('words.pkl','rb'))

classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    # define the sentence variable
    sentence = "Hello there!"

    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    #print(sentence_words)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    #print(sentence_words)

    # Remove the return statement
    return sentence_words


def bow(sentence, words, show_details=True):
    #tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    #print(sentence_words)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    #print(bag)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    #print(bag)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    #print(p)
    res = model.predict(np.array([p]))[0]
    #print(res)
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    #print(results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    #print(results)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    #print(return_list)
    return return_list


def getResponse(ints, intents_json):
    #print(ints)
    #print(intents_json)
    tag = ints[0]['intent']
    #print(tag)
    list_of_intents = intents_json['intents']
    #print(list_of_intents)
    for i in list_of_intents:
        #print(i)
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(text):
    ints = predict_class(text, model)
    #print(ints)
    res = getResponse(ints, intents)
    #print(res)
    return res

start = True

while start:
    query = input("Enter Message: ")
    if query in ['quit','q','exit','bye']:
        start = False
        continue
    try:
        res = chatbot_response(query)
        print(res)
    except FileNotFoundError:
        print("Model file not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
