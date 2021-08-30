from chat_bot import *


def main():
    bot = ChatBot()
    bot.parse_json('intents.json')
    bot.set_training_data()
    bot.set_default_model()
    bot.train_model(epochs=300, batch_size=20, verbose=1)

    bot.run()
    return 0


if __name__ == "__main__":
    main()
