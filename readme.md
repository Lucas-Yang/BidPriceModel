## 价格预测模型
### 文件说明
数据没有上传， 需要自己在根目录下创建data, data_res两个文件夹用于放数据

data:

├── xx.xlsx

├── xx.xlsx

├── xx.xlsx

└── xx.xlsx

data_res:

存放中间文件

data_generator.py： 包含两个类, DataEng，处理原始excel数据； DataGenerator, 基于处理好后的数据生成模型想要的数据

train.py: 训练入口
