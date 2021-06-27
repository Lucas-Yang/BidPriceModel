import os
import json
import pandas as pd
import numpy as np


class DataGenerator(object):
    """ 二次加工数据
    """

    def __init__(self, file_path, res_file):
        """
        """
        self.file_path = file_path
        self.__df = pd.read_csv(self.file_path)
        self.res_file = res_file

    def __write_csv(self, df, obj_name, company_name):
        """
        :param df:
        :return:
        """
        df.to_csv('data_res/single_data/{}_{}_{}.csv'.
                  format(obj_name,
                         company_name,
                         self.file_path.split('/')[-1].split('.')[0]
                         ),
                  index=False)

    def pre_func_score(self, price_dict: dict):
        """ 老方案偏差计算， price(t+2)= price(t) * 0.6 + price(t-1) * 0.3 + price(t-2) * 0.1
            备注 T 代表一个一个月份
        :param price_dict:
        :return:
        """
        t_1_x_list = []
        t_2_x_list = []
        t_3_x_list = []
        main_code_list = []
        company_code_list = []
        year_list = []
        t_label_list = []
        mape_list = []
        new_dict = {}
        for batch_index, batch_price in price_dict.items():
            main_code = str(batch_price["main_code_index"]) + "_" + str(batch_price["company_name_index"])
            if main_code not in new_dict:
                new_dict[main_code] = [(batch_price["年"], batch_price["月"],
                                        batch_price["agv_price"])]
            else:
                new_dict[main_code].append((batch_price["年"], batch_price["月"],
                                            batch_price["agv_price"]))
        print("##" * 20, "new-dict", "##" * 20)
        print(new_dict)
        num_main_code_useless = 0
        num_main_code_useful = 0
        for main_code, main_code_price_list in new_dict.items():
            if len(main_code_price_list) <= 2:
                num_main_code_useless += 1
            else:
                num_main_code_useful += 1
                main_code_price_list.sort()
                for batch_index, batch_price_tuple in enumerate(main_code_price_list):
                    batch_price = batch_price_tuple[2]

                    if batch_index == 0:
                        continue
                    elif 1 <= batch_index <= 2:
                        predict_batch_price = main_code_price_list[0][2]
                        main_code_list.append(main_code.split('_')[0])
                        company_code_list.append(main_code.split('_')[1])
                        year_list.append(main_code_price_list[batch_index][0] - main_code_price_list[0][0])
                        t_1_x_list.append(main_code_price_list[0][2])
                        t_2_x_list.append(main_code_price_list[0][2])
                        t_3_x_list.append(main_code_price_list[0][2])
                        t_label_list.append(batch_price)
                    elif batch_index == 3:
                        predict_batch_price = (main_code_price_list[1][2] * 0.6 +
                                               main_code_price_list[0][2] * 0.3) / 0.9

                        main_code_list.append(main_code.split('_')[0])
                        company_code_list.append(main_code.split('_')[1])
                        year_list.append(
                            (main_code_price_list[batch_index][0] - main_code_price_list[0][0]
                             +
                             main_code_price_list[batch_index][0] - main_code_price_list[1][0]
                             ) / 2
                        )
                        t_1_x_list.append(main_code_price_list[1][2])
                        t_2_x_list.append(main_code_price_list[0][2])
                        t_3_x_list.append(main_code_price_list[0][2])
                        t_label_list.append(batch_price)
                    elif batch_index > 3:
                        # print(main_code_price_list)
                        predict_batch_price = main_code_price_list[batch_index - 2][2] * 0.6 + \
                                              main_code_price_list[batch_index - 3][2] * 0.3 + \
                                              main_code_price_list[batch_index - 4][2] * 0.1

                        main_code_list.append(main_code.split('_')[0])
                        company_code_list.append(main_code.split('_')[1])

                        year_list.append(
                            (main_code_price_list[batch_index][0] - main_code_price_list[batch_index - 2][0]
                             +
                             main_code_price_list[batch_index][0] - main_code_price_list[batch_index - 3][0]
                             +
                             main_code_price_list[batch_index][0] - main_code_price_list[batch_index - 4][0]
                             ) / 3
                        )

                        t_1_x_list.append(main_code_price_list[batch_index - 2][2])
                        t_2_x_list.append(main_code_price_list[batch_index - 3][2])
                        t_3_x_list.append(main_code_price_list[batch_index - 4][2])
                        t_label_list.append(batch_price)
                    else:
                        print(batch_index)
                        continue

                    # 某些物料波动比较大，待v2版本重新处理数据，当前版本直接丢弃
                    # if abs((predict_batch_price - batch_price) / batch_price) >= 1:
                    #    continue
                    # print(batch_price)
                    if batch_price == 0:
                        continue
                    mape_list.append(abs((predict_batch_price - batch_price) / batch_price))

        print("usefull_code: ", num_main_code_useful,
              "useless_code: ", num_main_code_useless)
        print("#" * 20, "mape", "#" * 20)
        print(sum(mape_list) / len(mape_list))

        df = pd.DataFrame({"t-2": t_1_x_list,
                           "t-3": t_2_x_list,
                           "t-3": t_3_x_list,
                           "main_code": main_code_list,
                           "year_list": year_list,
                           "company_code": company_code_list,
                           "y": t_label_list
                           }
                          )
        df.to_csv(self.res_file, index=False)

    def pick_data(self, obj_name_list=None, company_name=None):
        """
        :return:
        """
        single_data_df = self.__df
        # single_data_df.drop(["含税单价", '采购数量'], axis=1, inplace=True)
        tmp_data = single_data_df.groupby(['main_code_index', "年", "月", "company_name_index"])['agv_price'].mean(). \
            reset_index(level=None,
                        drop=False,
                        name=None,
                        inplace=False
                        )
        tmp_data.to_csv('data_res/temp_data.csv')
        print(tmp_data.to_dict('index'))
        self.pre_func_score(tmp_data.to_dict('index'))
        # self.__write_csv(single_data_df, obj_name, company_name)

    def pick_obj(self):
        """
        :param:
        :return:
        """
        print(self.__df['MainCode'].value_counts())


class DataEng(object):
    """ 原始数据处理以及特征工程
    """

    def __init__(self, file_name_path=None):
        self.file_name_dict = file_name_path
        if file_name_path:
            self.data_frame = self.__read_xsl_file()
        else:
            self.data_frame = None

    def __read_xsl_file(self):
        """
        :return:
        """
        data_list = []
        for root_path, _, file_list in os.walk(self.file_name_dict):
            for file_name in file_list:
                data_list.append(pd.read_excel('./{}/{}'.format(root_path, file_name)))
        return pd.concat(data_list)

    def __write_json(self, json_data, res_file_name):
        """
        :param json_data:
        :param res_file_name:
        :return:
        """
        with open(res_file_name, "w") as file_obj:
            json.dump(json_data, file_obj, ensure_ascii=False)

    def __date_eng(self, df):
        """
        :param df:
        :return:
        """
        season_dict = {
            1: '春季', 2: '春季', 3: '春季',
            4: '夏季', 5: '夏季', 6: '夏季',
            7: '秋季', 8: '秋季', 9: '秋季',
            10: '冬季', 11: '冬季', 12: '冬季',
        }
        df["年"] = df['计划管理统计月份'].apply(lambda x: str(x)[0:4])
        df['月'] = df['计划管理统计月份'].apply(lambda x: int(str(x)[4:]))
        df['季节'] = df['月'].map(season_dict)

    def data_format(self):
        """
        :return:
        """
        company_info_dict = {}
        main_code_dict = {}
        self.data_frame.replace(np.inf, None, inplace=True)
        self.data_frame = self.data_frame.dropna(axis=0, how='any')
        self.data_frame.to_csv('./data_res/total.csv')
        self.data_frame.drop(["税率", "含税总价", "采购方式"], axis=1, inplace=True)
        self.data_frame["采购数量"] = self.data_frame["采购数量"].astype(int)
        self.data_frame["agv_price"] = self.data_frame.groupby(['MainCode', "计划管理统计月份", "单位编码", "单位名称"])['含税单价']. \
            transform('mean')
        self.data_frame["total_num"] = self.data_frame.groupby(['MainCode', "计划管理统计月份", "单位编码", "单位名称"])['采购数量']. \
            transform('sum')
        print(self.data_frame["agv_price"])
        self.data_frame["agv_price"] = self.data_frame["agv_price"].astype(int)

        self.data_frame.drop(["含税单价", '采购数量'], axis=1, inplace=True)
        self.data_frame.drop_duplicates(ignore_index=True, inplace=True, keep=False)
        self.__date_eng(self.data_frame)

        print(self.data_frame.head)
        ratio = (self.data_frame.isnull().sum() / len(self.data_frame)).sort_values(ascending=False)
        print(ratio)

        company_data = self.data_frame["单位名称"] + ":" + self.data_frame["单位编码"].map(str)
        for index, company_name in enumerate(set(self.data_frame["单位名称"]), start=1):
            company_info_dict[company_name] = index

        self.__write_json(company_info_dict, "data_res/company_info.json")

        for index, main_code in enumerate(set(self.data_frame["MainCode"]), start=1):
            main_code_dict[main_code] = index
        self.__write_json(main_code_dict, "data_res/main_code.json")

        self.data_frame["company_name_index"] = self.data_frame["单位名称"].map(company_info_dict)
        self.data_frame["main_code_index"] = self.data_frame["MainCode"].map(main_code_dict)
        self.data_frame["season_index"] = self.data_frame["季节"].map({"春季": 1, "夏季": 2, "秋季": 3, "冬季": 4})
        self.data_frame["year_index"] = self.data_frame["年"].map({"2014": 1, "2015": 2, "2016": 3, "2017": 4,
                                                                  "2018": 5, "2019": 6, "2020": 7, "2021": 8})
        self.data_frame["count_index"] = self.data_frame["月"].map({1: 1, 2: 1, 3: 2, 4: 2,
                                                                   5: 3, 6: 3, 7: 4, 8: 4,
                                                                   9: 5, 10: 5, 11: 6, 12: 6
                                                                   })

        train_data = self.data_frame[self.data_frame['计划管理统计月份'] < 201901]
        eval_data = self.data_frame[self.data_frame['计划管理统计月份'] >= 201901]
        train_data.to_csv('data_res/train_data.csv', index=False)
        eval_data.to_csv('data_res/eval_data.csv', index=False)


if __name__ == '__main__':
    # data_eng = DataEng('./data')
    # data_eng.data_format()

    data_handler = DataGenerator('data_res/train_data.csv', res_file='temp_data_train.csv')
    # data_handler.pick_obj()
    data_handler.pick_data()

    data_handler = DataGenerator('data_res/eval_data.csv', res_file='temp_data_test.csv')
    # data_handler.pick_obj()
    data_handler.pick_data()
