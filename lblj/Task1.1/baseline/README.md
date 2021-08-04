

本项目为 **中国法研杯司法人工智能挑战赛（CAIL2021） 论辩理解**赛道 任务1.1 **案件类型不敏感的争议观点对抽取** 的基线模型和参考代码，该模型是基于BERT的神经网络模型。

### 0. 预处理


#### 0.1 安装依赖包

我们使用的环境为`python 3.6.13`。

```
pip install -r requirements.txt
```

#### 0.2 数据集

数据集下载请访问比赛[主页](http://cail.cipsc.org.cn/)。

本项目中只使用了 `SMP-CAIL2021-train.csv`： 包含了2449对裁判文书中的互动论点对。分别包含以下维度：

  - `id`： 论点对id
  - `text_id`： 裁判文书id
  - `sc`： 论点对中诉方论点
  - `A/B/C/D/E`： 给出的五句候选辩方论点
  - `answer`： 辩方正确论点

划分训练集、验证集：

```
python prepare.py
```

#### 0.3 下载BERT模型（pytorch版本）

自行下载中文预训练BERT模型，存放于`model/bert`和`model/bert/bert-base-chinese`目录

中文预训练BERT模型包含三个文件：

1. [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json) 
2. [`pytorch_model.bin`](https://cdn.huggingface.co/bert-base-chinese-pytorch_model.bin)
3. [`vocab.txt`](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt) 

初始文件目录：

```
├── config
│   └── bert_config.json
├── data
│   ├── SMP-CAIL2021-train.csv
│   ├── train.csv
│   └── valid.csv
├── model
│   └── bert
│       ├── bert-base-chinese
│       │   ├── config.json
│       │   └── pytorch_model.bin
│       └── vocab.txt
├── __init__.py
├── data.py
├── evaluate.py
├── main.py
├── model.py
├── prepare.py
├── train.py
├── utils.py
└── requirements.txt
```

### 1. 训练

#### 1.1 BERT训练

采用4张2080Ti训练，训练参数可在`config/bert_config.json`中调整。

```
python -m torch.distributed.launch train.py --config_file 'config/bert_config.json'
```

<div align = "center">
  <img src="images/bert_train.png" width = "50%"/>
</div>


#### 1.2 训练成果

训练完成后文件目录：

`config`中包含模型训练参数。

`log`中包含模型每个epoch的Accuracy，F1 Score和每步loss的记录数据。

`model`中包含每个epoch训练后的模型和验证集上F1 Score最高的模型`model.bin`。

```
├── config
│   └── bert_config.json
├── data
│   ├── SMP-CAIL2021-train.csv
│   ├── train.csv
│   └── valid.csv
├── log
│   ├── BERT-epoch.csv
│   └── BERT-step.csv
├── model
│   └── bert
│       ├── BERT
│       │   ├── bert-1.bin
│       │   ├── bert-2.bin
│       │   ├── bert-3.bin
│       │   ├── bert-4.bin
│       │   ├── bert-5.bin
│       │   ├── bert-6.bin
│       │   ├── bert-7.bin
│       │   ├── bert-8.bin
│       │   ├── bert-9.bin
│       │   └── bert-10.bin
│       ├── model.bin
│       ├── bert-base-chinese
│       │   ├── config.json
│       │   └── pytorch_model.bin
│       └── vocab.txt
├── __init__.py
├── data.py
├── evaluate.py
├── main.py
├── model.py
├── prepare.py
├── train.py
├── utils.py
└── requirements.txt
```

### 2. 模型预测

`in_file`为待测试文件，`out_file`为输出文件。

```
python main.py --model_config 'config/bert_config.json' \
               --in_file 'SMP-CAIL2021-train.csv' \
               --out_file 'bert-train-1.csv'
```
```
python main.py --model_config 'config/bert_config.json' \
               --in_file 'SMP-CAIL2021-test1.csv' \
               --out_file 'bert-submission-test-1.csv'
```

### 3. 评估结果

`golden_file`为带真实答案标签的文件，`predict_file`为待测试模型生成的结果文件。第一阶段仅提供训练集`SMP-CAIL2021-train.csv`的真实答案标签，测试集`SMP-CAIL2021-test1.csv`的真实答案标签暂不提供。运行结果输出Accuracy和F1分数。

```
python evaluate.py --golden_file 'data/SMP-CAIL2021-train.csv' \
                   --predict_file 'bert-train-1.csv'
```
