from model import BidPriceModel


def train():
    model_handler = BidPriceModel()
    # model_handler.train_linear_model('temp_data_train.csv', 'temp_data_train.csv')
    model_handler.train_lgb_new('temp_data_train.csv', 'temp_data_test.csv')
    # model_handler.train_lstm_model('temp_data_train.csv', 'temp_data_test.csv')


if __name__ == '__main__':
    train()
