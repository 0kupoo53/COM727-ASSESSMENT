#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import joblib


lemmatizer = WordNetLemmatizer()

with open('osk.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = joblib.load('chatbot_model.pkl')

# return sentance as list of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Converts list of words into array confirming any words in wordlist
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to use model to confirm probabilty that user input is from each class and return as a list 
def predict_class(sentence):
    #convert sentance into vector showing words present in word list
    bow = bag_of_words(sentence) 
    # predict the probabailty of user sentance being from each class using trained model
    res = model.predict_proba(np.array([bow]))[0] 
    # discard all potential all classes where the proability is so low a default answer should be returned
    ERROR_THRESHOLD = 0.20
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sorts remaining classes by probability size
    results.sort(key=lambda x: x[1], reverse=True)
    #create a list to store possible classes with their probilities highest to lowest
    results_list = []
    # select highest class probability
    for r in results:
        results_list.append({'intent': classes[r[0]], 'probability': str(r[1])})   
    return results_list


# function to identify highest proabiity class and return an answer.
def get_response(intents_list, intents_json):
    # select first item on results list.  They are sorted highest to lowest so this will be highest probabilty answer
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    # select random response from class indentified
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    # return selected response
    return result

def predict_intent(text):
    # convert to lower case for comparison
    text = text.lower()
    # predict the class
    ints = predict_class(text)
    # if no class matched then write apology message with 'fallback' as tag. Otherwise return highest probabilty response with class
    if len(ints) == 0:
        tag = "fallback"
        res = "Sorry, I'm not sure I know that."
    else:
        tag = ints[0]['intent']
        res = get_response(ints, intents)
    return tag, res
# In[ ]:





# In[ ]:




