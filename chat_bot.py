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


class ChatBot:

    def __init__(self):
        """Initializes handlers and bot itself

        Args:
            (for future) add new handlers and maybe types of bot
        """
        self._lemmatizer = WordNetLemmatizer()  # expand options in future
        self._stemmer = PorterStemmer()
        self._ignored_letters = ['?', '.', ',']

    def parse_json(self, file_name: str) -> None:
        """Parses .json file, cleans up patterns and retrieves vocabulary

        Args:
            file_name: str
        """
        self._data = json.loads(open(file_name).read())
        self._vocabulary = []

        for i, intent in enumerate(self._data['intents']):
            cleaned_up_pattern = []
            for pattern in intent['patterns']:
                pattern = word_tokenize(pattern)
                cleaned_up_sent = [self._stemmer.stem(word.lower()) for word in pattern
                                   if word not in self._ignored_letters]
                self._vocabulary.extend(cleaned_up_sent)
                cleaned_up_sent = " ".join(cleaned_up_sent)
                cleaned_up_pattern.append(cleaned_up_sent)
            self._data['intents'][i]['cleaned_up_patterns'] = cleaned_up_pattern
        self._vocabulary = sorted(set(self._vocabulary))

    def create_bag_of_words(self, pattern: str) -> list:
        """Creates "bag-of-words" model of pattern/sentence based on current vocabulary

        Args:
            pattern: sentence to transform
        """
        bag = [0] * len(self._vocabulary)
        pattern = word_tokenize(pattern)
        pattern = [self._lemmatizer.lemmatize(word.lower()) for word in pattern if word not in self._ignored_letters]
        for i, word in enumerate(self._vocabulary):
            if self._lemmatizer.lemmatize(word.lower()) in pattern:
                bag[i] += pattern.count(word)
        return bag

    def set_training_data(self) -> tuple:
        """Creates training data and returns shapes of input and output data for custom model
        """
        training_data = []

        for i, intent in enumerate(self._data['intents']):
            output_class = [0] * len(self._data['intents'])
            output_class[i] += 1
            for cl_u_pattern in intent['cleaned_up_patterns']:
                training_data.append([self.create_bag_of_words(cl_u_pattern), output_class])

        random.shuffle(training_data)

        input_data = []
        output_data = []
        for sample in training_data:
            input_data.append(sample[0])
            output_data.append(sample[1])

        self._input_data = input_data
        self._output_data = output_data

        return (len(input_data[0]),), len(output_data[0])

    def set_default_model(self) -> None:
        """Sets default keras model:
            input -> dense-128-relu -> dense-64-relu -> output-softmax

            optimizer: SGD(lr=0.1, momentum=0.9, nesterov=True)
            Loss: categorical_crossentropy
            Metrics: accuracy

                Args:
                    input_data: list(bag of words)
                    output_data: list(class)
                """
        model = Sequential()
        model.add(Dense(128, input_shape=(len(self._input_data[0]),), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(self._output_data[0]), activation='softmax'))
        sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.set_model(model)

    def set_model(self, model) -> None:
        """Set Keras model
        """
        self._model = model

    def train_model(self, epochs, batch_size, verbose) -> None:
        """Trains Keras model

        Args:
            epochs: int
            batch_size: int
            verbose: int
        """
        self._model.fit(np.array(self._input_data), np.array(self._output_data), epochs=epochs,
                        batch_size=batch_size, verbose=verbose)

    def run(self) -> None:
        pass







def run_chatbot():
    tags_patterns, responses = parse_json('intents.json')
    dictionary = create_dict(tags_patterns)
    training_data = get_training_data(tags_patterns, dictionary)
    input_data = [input[0] for input in training_data]
    output_data = [output[1] for output in training_data]
    print(input_data[0])
    print(output_data)

    model = Sequential()
    model.add(Dense(128, input_shape=(len(input_data[0]),), activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.3))
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
    intents = json.loads(open('intents.json').read())  # change later

    tags_patterns = []
    responses = []

    for intent in intents['intents']:
        tags_patterns.append([intent['tag'], intent['patterns']])
        # patterns.append(intent['patterns'])
        responses.append(intent['responses'])

    return tags_patterns, responses


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
