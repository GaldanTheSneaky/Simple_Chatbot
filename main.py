from chat_bot import *
import requests
import pandas as pd

def custom_response(message):
    df = pd.read_csv('worldcities.csv', sep=',')
    message = word_tokenize(message)
    location = ""
    for word in message:
        count = (df['city'] == word).sum()
        if count > 0:
            location = word

    url = 'http://api.openweathermap.org/data/2.5/weather?q=' + location + '&APPID=6622384c4378ac8cc331e472f22b9e96'
    r = requests.get(url)
    data = r.json()
    response = "The temperature in " + location + " now is " + str(int(data['main']['temp'] - 273.15)) \
               + " degrees celcius"

    return response

def main():
    bot = ChatBot()
    bot.parse_json('intents.json')
    bot.set_training_data()
    model = keras.models.load_model("chatbot_model")
    bot.set_model(model)
    bot.set_behavior('goodbye', terminate=True)
    bot.set_behavior('weather', behavior=custom_response)
    bot.run()

    return 0


if __name__ == "__main__":
    main()
