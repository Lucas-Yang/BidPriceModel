import os
import json
import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Bidirectional, Conv1D


class BidPriceModel(object):
    """ 中标价格预测类
    """

    def __init__(self):
        pass

    def train_lgb(self, data_file_path):
        """
        :return:
        """
        df = pd.read_csv(data_file_path)
        x_df = df[["company_name_index", "year_index", "count_index", "season_index", "main_code_index"]]
        # train_x = x_df[:int(len(x_df)*0.2)]
        # test_x = x_df[int(-len(x_df)*0.2):]
        train_x = x_df
        test_x = x_df[int(-len(x_df) * 0.1):]

        y_df = df["agv_price"]
        train_y = y_df
        test_y = y_df[int(-len(x_df) * 0.1):]
        # train_y = y_df[:int(-len(x_df)*0.2)]
        # test_y = y_df[int(-len(x_df)*0.2):]

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'regression',  # 目标函数
            'metric': {'rmse'},  # 评估函数
            'num_leaves': 31,  # 叶子节点数
            'learning_rate': 0.05,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.8,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1
        }
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)
        gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_eval, early_stopping_rounds=50)
        gbm.save_model('model.txt')
        y_pred = gbm.predict(test_x, num_iteration=gbm.best_iteration)
        # 评估模型
        print('The rmse of prediction is:', mean_squared_error(test_y, y_pred) ** 0.5)
        print(pd.DataFrame({
            'column': x_df.columns,
            'importance': gbm.feature_importance(),
        }).sort_values(by='importance'))

        """
        gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
        gbm.fit(train_x, train_y, eval_set=[(test_x, test_y)], eval_metric='l1', early_stopping_rounds=5)
        # 测试机预测
        y_pred = gbm.predict(test_x, num_iteration=gbm.best_iteration_)
        # 模型评估
        print('The rmse of prediction is:', mean_squared_error(test_y, y_pred) ** 0.5)
        # feature importances
        print('Feature importances:', list(gbm.feature_importances_))
        """

    def train_lgb_new(self, train_data_path, test_data_path):
        """
        :return:
        """
        df = pd.read_csv(train_data_path)
        df_test = pd.read_csv(test_data_path)

        train_x = df[["t-2", "t-3", "t-4"]]
        train_y = df["y"]

        test_x = df_test[["t-2", "t-3", "t-4"]]
        test_y = df_test["y"]

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'regression',  # 目标函数
            'metric': {'mape'},  # 评估函数
            'num_leaves': 31,  # 叶子节点数
            'learning_rate': 0.05,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.8,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1
        }
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)

        gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_eval, early_stopping_rounds=50)
        gbm.save_model('model.txt')
        y_pred = gbm.predict(test_x, num_iteration=gbm.best_iteration)
        # 评估模型
        print('The rmse of prediction is:', mean_squared_error(test_y, y_pred) ** 0.5)
        print(pd.DataFrame({
            'column': train_x.columns,
            'importance': gbm.feature_importance(),
        }).sort_values(by='importance'))

    def train_linear_model(self, train_data_path, test_data_path):
        """
        :return:
        """
        df = pd.read_csv(train_data_path)
        df_test = pd.read_csv(test_data_path)

        train_x = df[["t-1", "t-2", "t-3"]]
        train_y = df["y"]

        test_x = df_test[["t-1", "t-2", "t-3"]]
        test_y = df_test["y"]

        model = LinearRegression()
        model.fit(train_x, train_y)

        predict_y = model.predict(test_x)

        print(predict_y)
        n = len(test_y)
        mape = sum(np.abs((test_y - predict_y) / test_y)) / n

        print(mape)
        print(model.coef_)  # 系数，有些模型没有系数（如k近邻）

    def train_lstm_model(self, train_data_path, test_data_path):
        """
        :param
        :return:
        """
        df = pd.read_csv(train_data_path)
        df_test = pd.read_csv(test_data_path)
        # df = shuffle(df)
        # train_x = df[["company_name_index", "year_index", "月", "season_index", "main_code_index", "count_index"]]
        # train_y = df["agv_price"]
        train_x = df[["t-2", "t-3", "t-4"]]
        train_y = df["y"]
        # df_norm = (y_df - y_df.min()) / (y_df.max() - y_df.min())
        # print(df_norm)
        train_x, train_y = np.array(train_x), np.array(train_y)
        # train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

        # test_x = df_test[["company_name_index", "year_index", "月", "season_index", "main_code_index", "count_index"]]
        # test_y = df_test["agv_price"]
        test_x = df_test[["t-2", "t-3", "t-4"]]
        test_y = df_test["y"]

        test_x, test_y = np.array(test_x), np.array(test_y)
        # test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

        print(train_x.shape)
        model = Sequential()
        # model.add(LSTM(units=100, return_sequences=True, input_shape=(train_x.shape[1], )))
        # model.add(Dropout(0.5))

        model.add(Dense(64, activation='relu', input_shape=(train_x.shape[1], )))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mape', optimizer='adam', metrics=['mape'])
        model.summary()
        checkpoint = ModelCheckpoint('best_deep_model1.h5', monitor='val_mape', verbose=1,
                                     save_best_only=True, mode='min', period=1)

        model.fit(train_x, train_y, epochs=100, batch_size=8,
                  validation_data=(test_x, test_y), callbacks=[checkpoint])

    def eval(self, eval_data_path=None):
        """
        :return:
        """
        df = pd.read_csv(eval_data_path)
        df = shuffle(df)
        test_x = df[["company_name_index", "year_index", "月", "season_index", "main_code_index"]]
        test_y = df["agv_price"]
        bsm = lgb.Booster(model_file='model.txt')
        pred_y = bsm.predict(test_x)
        print(pred_y)
        print(test_y)
        print('在测试集上的rmse为:')
        print(mean_squared_error(test_y, pred_y) ** 0.5)


if __name__ == '__main__':
    # model_handler = BidPriceModel('./data')
    # model_handler.data_format()
    model_handler = BidPriceModel()
    model_handler.train_lstm_model('data_res/train_data.csv', 'data_res/eval_data.csv')
    # model_handler.train_lgb('data_res/train_data.csv')
    # model_handler.eval('data_res/eval_data.csv')
