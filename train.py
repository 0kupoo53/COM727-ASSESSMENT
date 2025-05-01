import random
import json
import pickle
import numpy as np
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier


nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# define lemitiser to reduce all words to common simplier form
lemmatizer = WordNetLemmatizer()

# load training data
# Open and load the JSON file with proper encoding
with open('osk.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# create empty lists to recieve data
words = []
classes = []
documents = []

# define puncuation and common 'stop' words we do not wish to train on.
ignore_letters=['?','!','.','/','@']
stop_words = set(stopwords.words('english'))
# review each sentance in each class and transform into list of words, classes and words associated with each class
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # split sentance into separate words
        word_list = nltk.word_tokenize(pattern)
        # reduce words to simpler form and discarding punctuation
        word_list = [lemmatizer.lemmatize(word) for word in word_list if word not in ignore_letters]
        # reduce all words to lower case and discarding common 'stop' words
        word_list = [word.lower() for word in word_list if word.lower() not in stop_words]
        # add words from this sentance to master 'words' list 
        words.extend(word_list)
        # add words in word list master class/words document
        documents.append((word_list,intent['tag']))
        # add class name if new
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# remove duplicates and sort words alphibetically
words = sorted(set(words))

# sort classes alphibetically
classes = sorted(set(classes))

# store list of words from training set intents
pickle.dump(words, open('words.pkl', 'wb'))
# store classes/tags
pickle.dump(classes, open('classes.pkl', 'wb'))

# create an empty list to store training data for model training
training = []

# store words in each class in 'training'
for document in documents:
    bag=[]
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = classes.index(document[1])  
    training.append([bag, output_row])

# create empty lists to store classes and words in training for that class
train_x = []
train_y = []

# store classes and words in training for that class from 'training'
for item in training:
    train_x.append(item[0])  
    train_y.append(item[1])  

# Convert train_x and train_y to arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

# Use nureal network model to train data
model = MLPClassifier(max_iter=400, random_state=42)
hist = model.fit(np.array(train_x), np.array(train_y))

# Save model for use in chatbot
joblib.dump(model, 'chatbot_model.pkl')

# Confirm training and save done to user
print("The model is trained and saved.")