
本项目为 **中国法研杯司法人工智能挑战赛（CAIL2021） 论辩理解**赛道 任务2.1 **争议点类型识别** 的基线模型和参考代码，该模型是基于BERT多标签分类模型和随机森林多标签分类模型的基线系统。

### 0. 预处理


#### 0.1 安装依赖包

我们使用的环境为`python 3.7.10`。

```
pip install -r requirements.txt
```

#### 0.2 数据集

数据集下载请访问比赛[主页](http://cail.cipsc.org.cn/)。

本项目中只使用了 `SMP-CAIL2021-focus_recognition-train.json`： 包含了168篇裁判文书数据。分别包含以下维度：

  - `文书ID`： 裁判文书id
  - `诉称段`： 裁判文书的诉称段内容
  - `辩称段`： 裁判文书的辩称段内容
  - `裁判分析过程段`： 裁判文书的裁判分析过程段内容
  - `争议焦点`： 裁判文书对应的争议焦点
  - `诉讼请求`： 裁判文书对应的诉讼请求
  - `抗辩事由`： 裁判文书对应的抗辩事由
  - `争议焦点子节点`： 裁判文书中标注的争议焦点子节点
  - `文书段落`： 用于说明标注信息来源于裁判文书中对应的段落
  - `要素名称`： 所需要识别的争议焦点、诉讼请求、抗辩事由以及争议焦点子节点的名称
  - `oValue`： 标注信息来源的文字范围（非一定是整句， 可能是一个词汇或者其他粒度）
  - `sentence`： 标注信息来源的句子（必为整句）



#### 0.3 下载BERT模型（tensorflow版本）

自行下载中文预训练BERT模型，存放于`step1_model/BERT_BASE_DIR`目录

中文预训练BERT模型包含5个文件：

1. bert_config.json
2. bert_model.ckpt.data-00000-of-00001
3. bert_model.ckpt.index
4. bert_model.ckpt.meta
5. vocab.txt

下载地址[here](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

初始文件目录：

```
├── data
│   └── raw_data
│       └── SMP-CAIL2021-focus_recognition-train.json
├── step1_model
│   ├── bert
│   │   ├── __init__.py
│   │   ├── bert_predict.py
│   │   ├── modeling.py
│   │   ├── optimization.py
│   │   ├── run_classifier_multi_label.py
│   │   ├── tokenization.py
│   │   └── train_model.sh
│   └── BERT_BASE_DIR
│       ├── bert_config.json
│       ├── bert_model.ckpt.data-00000-of-00001
│       ├── bert_model.ckpt.index
│       ├── bert_model.ckpt.meta
│       └── vocab.txt
├── step2_model
│   ├── __init__.py
│   ├── data_util.py
│   ├── model_predict.py
│   └── model_train.py
├── __init__.py
├── pipeline_evaluate.py
├── pipeline_predict.py
├── prepare.py
├── prepare_after_step1.py
├── prepare.py
├── utils.py
└── requirements.txt
```



### 1. 模型训练及数据集构造与生成

#### 1.0 数据集（训练step1_model之前）

根据原始数据集，划分第一步模型以及全流程的的训练集、验证集和标签等文件：

```
python prepare.py   --src_path "./data/raw_data/SMP-CAIL2021-focus_recognition-train.json" \
                    --single_file_train_dataset_path "data/single_file_data/train_data.txt" \
                    --single_file_test_dataset_path "data/single_file_data/test_data.txt" \
                    --step1_train_data_path "data/step1_data/train_data.txt" \
                    --step1_test_data_path "data/step1_data/test_data.txt" \
                    --label_list_path "data/step1_data/labels.txt"
```

#### 1.1 step1_model训练(BERT)

在`step1_model/bert`中运行如下命令，相关参数可以在`train_model.sh`中直接修改。

```
sh train_model.sh
```
#### 1.2 step2_model数据集生成(step1模型完成，训练step2_model之前)

根据step1_model训练好的模型以及`data/single_file_data`下的数据生成step2_model的训练和测试数据集。

当step1模型训练好之后，运行如下命令。

```
python prepare_after_step1.py   --train_dataset_path "data/single_file_data/train_data.txt" \
                                --test_dataset_path "data/single_file_data/test_data.txt" \
                                --step2_train_dataset_path "data/step2_data/train_data.txt" \
                                --step2_test_dataset_path "data/step2_data/test_data.txt" \
                                --label_list_path "data/step2_data/labels.txt" \
                                --vocab_file './step1_model/BERT_BASE_DIR/vocab.txt' \
                                --bert_config_file './step1_model/BERT_BASE_DIR/bert_config.json' \
                                --ckpt_dir './step1_model/output' \
                                --max_seq_len 500 \
                                --n_classes 130
```

#### 1.3 step2_model训练(随机森林)

在`step2_model`中运行如下命令即可训练第二步模型。

```
python model_train.py    --train_data_path "../data/step2_data/train_data.txt" \
                         --test_data_path "../data/step2_data/test_data.txt" \
                         --label_list_path "../data/step2_data/labels.txt" \
                         --model_save_path "./output/model.pkl"

```


#### 1.4 训练成果

训练完成后文件目录：



```
├── data
│   ├── raw_data
│   │   └── SMP-CAIL2021-focus_recognition-train.json
│   ├── single_file_data
│   │   │   ├── test_data.txt
│   │   │   └── train_data.txt
│   ├── step1_data
│   │   │   ├── labels.txt
│   │   │   ├── test_data.txt
│   │   │   └── train_data.txt
│   └── step2_data
│           ├── labels.txt
│           ├── test_data.txt
│           └── train_data.txt
├── step1_model
│   ├── bert
│   │   ├── __init__.py
│   │   ├── bert_predict.py
│   │   ├── modeling.py
│   │   ├── optimization.py
│   │   ├── run_classifier_multi_label.py
│   │   ├── tokenization.py
│   │   └── train_model.sh
│   ├── BERT_BASE_DIR
│   │   ├── bert_config.json
│   │   ├── bert_model.ckpt.data-00000-of-00001
│   │   ├── bert_model.ckpt.index
│   │   ├── bert_model.ckpt.meta
│   │   └── vocab.txt
│   └── output
│       └── ...
├── step2_model
│   ├── output
│   │   └── model.pkl
│   ├── __init__.py
│   ├── data_util.py
│   ├── model_predict.py
│   └── model_train.py
├── __init__.py
├── pipeline_evaluate.py
├── pipeline_predict.py
├── prepare.py
├── prepare_after_step1.py
├── prepare.py
├── utils.py
└── requirements.txt
```

### 2. 模型预测

整个基线系统以pipeline形式将两个模型进行串联，模型的预测需要运行`pipeline_predict.py`将会对`data/single_file_data/test_data.txt`进行预测， 具体的命令如下。

```
python pipeline_predict.py  --label_list_path "./data/step2_data/labels.txt" \
                            --test_data_path "./data/single_file_data/test_data.txt" \
                            --step2_model_path "./step2_model/output/model.pkl" \
                            --output_prediction_path "./data/output/prediction.txt" \
                            --output_tag_path "./data/output/criterion.txt" \
                            --vocab_file './step1_model/BERT_BASE_DIR/vocab.txt' \
                            --bert_config_file './step1_model/BERT_BASE_DIR/bert_config.json' \
                            --ckpt_dir './step1_model/output' \
                            --max_seq_len 500 \
                            --n_classes 130
```

运行上述命令将会在`data/output`下生成`criterion.txt`和`prediction.txt`，分别代表`data/single_file_data/test_data.txt`的标准标签和基线系统预测的标签。



### 3. 评估结果

针对基线系统的预测输出和原始测试集标签进行评估，评估方法可见[主页](http://cail.cipsc.org.cn/task6.html?raceID=4)

```
python pipeline_evaluate.py --prediction_path "./data/output/prediction.txt" \
                            --label_path "./data/output/criterion.txt"
```

