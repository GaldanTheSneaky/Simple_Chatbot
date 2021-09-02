from chat_bot import *

def custom_response():
    return "i'm bot"

def main():
    bot = ChatBot()
    bot.parse_json('intents.json')
    bot.set_training_data()
    model = keras.models.load_model("chatbot_model")
    bot.set_model(model)
    bot.set_behavior('goodbye', terminate=True)
    bot.set_behavior('name', behavior=custom_response)
    bot.run()
    return 0


if __name__ == "__main__":
    main()
