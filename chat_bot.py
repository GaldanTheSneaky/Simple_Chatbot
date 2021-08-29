# preprocess initial sentences
# restructure with OOP

import json
import random
import numpy as np
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
ignored_letters = ['?', '.', ',']


def run_chatbot():
    tags_patterns, responses = parse_json('intents.json')
    dictionary = create_dict(tags_patterns)
    training_data = get_training_data(tags_patterns, dictionary)
    input_data = [input[0] for input in training_data]
    output_data = [output[1] for output in training_data]
    print(input_data[0])
    print(output_data)

    model = Sequential()
    model.add(Dense(128, input_shape= (len(input_data[0]),), activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(len(output_data[0]), activation='softmax'))

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(np.array(input_data), np.array(output_data), epochs=300, batch_size=20, verbose=1)
    model.save('chatbot_model.model')
    print('Done')

    while True:
        message = input('type something: ')
        message = create_bag_of_words(message, dictionary)
        result = model.predict(np.array([message]))[0]
        result_index = np.argmax(result)
        if result[result_index] > 0.8:
            response = random.choice(responses[result_index])
        else:
            response = "What are u talking about"

        print(response)





def get_training_data(tags_patterns, dictionary):
    input_data = []
    output_data = []
    classes = [tag[0] for tag in tags_patterns]

    for pattern in tags_patterns:
        for sentence in pattern[1]:
            input_data.append(create_bag_of_words(sentence, dictionary))
            output = [0] * len(tags_patterns)
            output[classes.index(pattern[0])] = 1
            output_data.append(output)

    training_data = list(zip(input_data, output_data))
    random.shuffle(training_data)

    return training_data


def parse_json(json_file):
    intents = json.loads(open('intents.json').read()) # change later

    tags_patterns = []
    responses = []

    for intent in intents['intents']:
        tags_patterns.append([intent['tag'], intent['patterns']])
        #patterns.append(intent['patterns'])
        responses.append(intent['responses'])

    return tags_patterns,  responses


def create_dict(tag_patterns):
    dictionary = []

    for pattern in tag_patterns:
        for sentence in pattern[1]:
            dictionary.extend(word_tokenize(sentence))

    dictionary = [lemmatizer.lemmatize(word.lower()) for word in dictionary if word not in ignored_letters]
    dictionary = sorted(set(dictionary))
    return dictionary


def create_bag_of_words(sentence, dictionary):
    bag = []
    sentence = word_tokenize(sentence)
    sentence = [lemmatizer.lemmatize(word.lower()) for word in sentence if word not in ignored_letters]
    for word in dictionary:
        bag.append(1) if lemmatizer.lemmatize(word.lower()) in sentence else bag.append(0)
    return bag
