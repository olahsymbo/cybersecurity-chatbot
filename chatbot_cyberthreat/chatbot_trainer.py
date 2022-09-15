import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

classes = ['phishing',
    'phishing_attack',
    'phishing_solution'
    'Malware',
    'Malware_attack',
    'Malware_solution',
    'Ransomware',
    'Ransomware_attack',
    'Ransomware_solution',
    'Man_in_the_middle',
    'Man_in_the_middle_attack',
    'Man_in_the_middle_solution',
    'Denial_Of_Service',
    'Denial_Of_Service_attack',
    'Denial_Of_Service_solution',
    'Crypto-Jacking',
    'Crypto-Jacking_attack',
    'Crypto-Jacking_solution',
    'not_cyberattack', 
    'goodbye',
    'greeting', 
    'noanswer',
    'options', 
    'thanks']


with open("classes.pkl", "wb") as output_file:
    pickle.dump(classes, output_file)

words=[]
classes = []
docs = []
ignore_letters = ['!', '?', ',', '.']
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern) #tokenize each word
        words.extend(word)
        docs.append((word, intent['tag']))#add documents in the corpus
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#---------- Lematization -----------#
            
# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes))) 

#---------- save the words and classes ------------#
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# create bag of words for each sentence
for doc in docs: 
    bag = []
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
X_train, y_train = list(training[:,0]), list(training[:,1])
print("Training data created")

# Create model - 4 layers CNN. 
text_model = Sequential()
text_model.add(Dense(256, input_shape=(len(X_train[0]),), activation='relu'))
text_model.add(Dense(128, activation='relu'))
text_model.add(Dropout(0.5))
text_model.add(Dense(64, activation='relu'))
text_model.add(Dropout(0.5))
text_model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile model. 
sgd = SGD(lr=0.01,  momentum=0.9, nesterov=True)
text_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = text_model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
text_model.save('chatbot_model.h5', hist)

print("model created")
