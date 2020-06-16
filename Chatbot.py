import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import numpy as np
import tensorflow as tf
import random
import nltk

import json
import pickle
from nltk.stem.lancaster import LancasterStemmer # to stem the words


#take the each word in the pattern to the base word such as whats to what
stemmer = LancasterStemmer()


# To read the datasets
with open('dataset.json') as file:
    dataset = json.load(file)


try:
    with open("train.pickle", "rb") as f:
        wordlist, label, training, result = pickle.load(f)
except:
    wordlist = []
    label = []
    patterns = [] #to label the pattern
    intention = [] # to classify the pattern

    for data in dataset['dataset']:
        for pattern in data['patterns']:
            words = nltk.word_tokenize(pattern) #get all the word in the pattern
            wordlist.extend(words)
            #add all the words into the wordlist rather than put the list inside the list
            patterns.append(words)
            intention.append(data['tag'])

            if data["tag"] not in label:
                label.append(data['tag'])

    #if word not in "?" is to remove the ? in the wordlist
    wordlist = [stemmer.stem(word.lower()) for word in wordlist if word != "?"] # to make all the base word to become lower case and ignore the question mark and exclamation mark
    wordlist = sorted(list(set(wordlist))) # to remove the duplicate elements (set()) in the wordlist and sort it

    #sorted tag (data['tag'])
    label = sorted(label)



    #training

    training = []
    result = []
    #contain the bag of word which is the sentence that have been encoded into integer

    empty = [0 for _ in range(len(label))]

    for index , value in enumerate(patterns):
        bagofwords = [] # the list that contain the representative of the word in numbers
        words = [stemmer.stem(wrds.lower()) for wrds in value] # to get the each letter

        for wrds in wordlist: #this wrds is each word in wordlist
            if wrds in words:
                #when the word is in the list then add 1 as the representative that the word is exist
                bagofwords.append(1)
            else: # other wise add 0 while the word does not exist
                bagofwords.append(0)

        copyofempty = empty.copy()
        #to get the copy of the empty list
        copyofempty[label.index(intention[index])] = 1
        #replace the empty list with index of the intention[the index of enumerate]


        training.append(bagofwords)
        # add the word that have been checked to be train
        result.append(copyofempty)
        # add the empty list to result

    training = np.array(training)
    result = np.array(result)

    with open("train.pickle", "wb") as f:
            pickle.dump((wordlist, label, training, result), f)

tf.reset_default_graph()

# building model for the bot

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(len(training[0]),)), #a fully connected layer that has been activated by Relu Layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(result[0]), activation='softmax')
])
sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

#to make sure that the is load whenever the model is used without pre-processing the data

try:
    model.load_weights('Our Bot')
except:
    model.fit(training, result, epochs=1000, batch_size=20)
    model.save("Our Bot")


def bagsofwords(sentence, wordlist):
    #create a blank bag wordlist

    bag_of_words = [0 for _ in range(len(wordlist))]

    breaksentence = nltk.word_tokenize(sentence)
    breaksentence = [stemmer.stem(word.lower()) for word in breaksentence]

    for letter in breaksentence:
        for index, value in enumerate(wordlist):
            if value == letter: # if the letter in bag_of_words is the same as wordlist
                bag_of_words[index] = 1

    return bag_of_words


def chat(type_message):
    reply = model.predict(np.array([bagsofwords(type_message , wordlist)])) #the probability
    reply_max = np.argmax(reply) #give the index with the largest value in reply
    tag = label[reply_max] #give the label according to the message

    for tags in dataset['dataset']:
        if tags['tag'] == tag:
            response = tags['responses']

    return random.choice(response)

