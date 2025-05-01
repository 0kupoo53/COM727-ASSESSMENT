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
print(f"Loaded model type: {type(model)}")
if hasattr(model, 'predict_proba'):
    print("The loaded model has a 'predict_proba' method.")
else:
    print("The loaded model DOES NOT have a 'predict_proba' method.")

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

def predict_class(sentence):
    #convert sentance into vector showing words present in word list
    bow = bag_of_words(sentence)

    #predict probabailty of each sentance being from each class using trained model
    res = model.predict_proba(np.array([bow]))[0]

    #removes all classes where the proability is less than 50%
    ERROR_THRESHOLD = 0.5
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # sorts remaining classes by probability size
    results.sort(key=lambda x: x[1], reverse=True)

    #create a list of classes with their probilities highest to lowest
    results_list = []
    for r in results:
        results_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return results_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Hi, I'm OSK, the school chatbot. How can I help?")

while True:
    message = input("You: ")
    ints = predict_class(message)
    #   if the probability of a sentance being from one of the classes is less than 50% ask user to re-phrase it.
    # Otherwise, return top probabilty response.
    if len(ints) == 0:
        res = "Sorry, I'm not sure I know that.  Please can you ask it another way."
    else:
        res = get_response(ints, intents)
    print(res)

# In[ ]:





# In[ ]:




