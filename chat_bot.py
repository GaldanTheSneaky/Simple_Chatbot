# consider creating Intent class
import json
import random
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


class ChatBot:

    def __init__(self):
        """Initializes handlers and bot itself

        Args:
            (for future) add new handlers and maybe types of bot
        """
        self._lemmatizer = WordNetLemmatizer()  # expand options in future
        self._stemmer = PorterStemmer()
        self._behavior_dict = {}
        self._vocabulary = []
        self._intention = -1  # lame, consider future fix
        self._ignored_letters = ['?', '.', ',']

    def parse_json(self, file_name: str) -> None:
        """Parses .json file, cleans up patterns and retrieves vocabulary and behavior for every intent

        Args:
            file_name: str
        """
        self._data = json.loads(open(file_name).read())

        for i, intent in enumerate(self._data['intents']):
            cleaned_up_pattern = []
            for pattern in intent['patterns']:
                self._behavior_dict[intent['tag']] = {'behavior': self.__give_default_response, 'terminate': False}
                pattern = word_tokenize(pattern)
                cleaned_up_sent = [self._stemmer.stem(word.lower()) for word in pattern
                                   if word not in self._ignored_letters]
                self._vocabulary.extend(cleaned_up_sent)
                cleaned_up_sent = " ".join(cleaned_up_sent)
                cleaned_up_pattern.append(cleaned_up_sent)
            self._data['intents'][i]['cleaned_up_patterns'] = cleaned_up_pattern
        self._vocabulary = sorted(set(self._vocabulary))

    def __give_default_response(self, user_message: str) -> str:
        """Gives random respond corresponding to intent
        """
        return random.choice(self._data['intents'][self._intention]['responses'])

    def __create_bag_of_words(self, pattern: str) -> list:
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
                training_data.append([self.__create_bag_of_words(cl_u_pattern), output_class])

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

            optimizer: SGD(lr=0.01, momentum=0.9, nesterov=True)
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
        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.set_model(model)

    def set_model(self, model) -> None:
        """Set Keras model
        """
        self._model = model

    def train_model(self, epochs, batch_size, verbose, save=False) -> None:
        """Trains Keras model

        Args:
            epochs: int
            batch_size: int
            verbose: int
            save: bool
        """
        self._model.fit(np.array(self._input_data), np.array(self._output_data), epochs=epochs,
                        batch_size=batch_size, verbose=verbose)

        if save:
            self._model.save('chatbot_model')

    def set_behavior(self, intent: str, behavior=None, terminate=False) -> None:
        """Sets custom behavior to intention

        Args:
            intent: intention
            behavior: custom function(user_message)
            terminate: set True if you want intention to shut down the bot
        """
        if behavior:
            self._behavior_dict[intent]["behavior"] = behavior
        self._behavior_dict[intent]["terminate"] = terminate

    def run(self) -> None:
        """ Runs bot until intention with terminate=True
            Bot gives proper answer with confidence > 85%
        """
        while True:
            message = input('type something: ')
            rendered_message = self.__create_bag_of_words(message)
            intent_probs = self._model.predict([rendered_message])[0]
            intent_idx = np.argmax(intent_probs)
            if intent_probs[intent_idx] > 0.85:
                self._intention = intent_idx
                print(self._behavior_dict[self._data['intents'][intent_idx]['tag']]['behavior'](message))
                if self._behavior_dict[self._data['intents'][intent_idx]['tag']]['terminate']:
                    break
            else:
                self._intention = -1
                print("Sorry, i don't understand you")


