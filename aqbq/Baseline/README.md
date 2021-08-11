# CAIL2021 —— 案情标签预测Baseline

该模型为使用bert-base-chinese进行预测的基线模型。

# 训练步骤：

1. 将训练数据拷贝至`data`下
2. 使用`python3 train.py`进行训练

# 提交步骤：

参考题目主目录中`python_sample/main/main.py`

你可以在`python_sample`中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个`zip`文件进行提交，该`zip`文件内部形式可以参看`python_sample/main.zip`。该`zip`文件**内部顶层**必须包含`main.py`，为运行的入口程序，我们会在该目录下使用`python3 main.py`来运行你的程序。

对于你的代码，你需要从`/input/`中读取数据进行预测。

在该文件夹中包含**若干**文件，每个文件均由若干行`json`格式数据组成。每行的数据格式与下发数据格式完全一致。选手需要将预测的结果输出到`/output/result.txt`中，以`json`格式输出一个列表。你需要按照测试集的顺序，依次输出每个测试样本所对应的标签，类型为`list`。（格式详见`Baseline/output/result.txt`）

以上为`main.py`中你需要实现的内容，你可以利用`python_sample`下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到`/work`路径下然后运行。

## 预训练语言模型

因评测环境不允许连接互联网，若选手需要使用预训练语言模型，则需要自己手动上传。评测环境中已经支持`transformers`中部分预训练语言模型，可直接通过 `model = AutoModel.from_pretrained(modelname)` 进行加载。已支持的预训练语言模型包括：

```
bert-base-chinese
thunlp/Lawformer
hfl/chinese-bert-wwm-ext
hfl/chinese-electra-180g-base-discriminator
hfl/chinese-legal-electra-base-discriminator
hfl/chinese-macbert-base
hfl/chinese-macbert-large
```

如有进一步需要可联系管理员进行安装。

