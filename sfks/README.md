# CAIL2021——司法考试

该项目为 **CAIL2021——司法考试** 的代码和模型提交说明。

数据集下载请访问比赛[主页](http://cail.cipsc.org.cn/)。

## 数据集引用

如果你要在学术论文中引用数据集，请使用如下bib

```tex
@article{zhong2019jec,
  title={JEC-QA: A Legal-Domain Question Answering Dataset},
  author={Zhong, Haoxi and Xiao, Chaojun and Tu, Cunchao and Zhang, Tianyang and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:1911.12011},
  year={2019}
}
```

## 选手交流群

QQ群：237633234

## 数据说明

本任务所使用的数据集来自于论文``JEC-QA: A Legal-Domain Question Answering Dataset``的司法考试数据集。

下发的文件包含``0_train.json,1_train.json``，分别对应概念理解题和情景分析题。

两个文件均包含若干行，每行数据均为json格式，包含若干字段：

- ``answer``：代表该题的答案。
- ``id``：题目的唯一标识符。
- ``option_list``：题目每个选项的描述。
- ``statement``：题干的描述。
- ``subject``：代表该问题所属的分类，仅有部分数据含有该字段。
- ``type``：无意义字段。

实际测试数据不包含``answer``字段。

## 提交的文件格式及组织形式

你可以在 ``python_sample`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``python_sample/main.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``/input/``中读取数据进行预测。

在该文件夹中包含**若干**文件，每个文件均由若干行``json``格式数据组成。每行的数据格式与下发数据格式完全一致。选手需要从将预测的结果输出到``/output/result.txt``中，以``json``格式输出一个字典。对于编号为``id``的题目，你需要在输出的字典中设置``id``字段，并且该字段内容为该题答案，类型为``list``。

以上为 ``main.py`` 中你需要实现的内容，你可以利用 ``python_example`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/work``路径下然后运行。

**请注意，如果你想要自己通过命令行运行python代码，请按照如下命令运行**

```bash
sudo /home/user/miniconda/bin/python3 work.py
```

## 其他语言的支持

如上文所述，我们现阶段只支持 ``python`` 语言的提交，但是这并不代表你不能够使用其他语言进行预测。你可以使用``python3 main.py``去调用运行其他语言的命令。但请注意，在你调用其他命令的时候请在命令前加上``sudo``以保证权限不会出问题。

## 现有的系统环境

| 软件名称 | 版本号 |
| -------- | ------ |
| python   | 3.8.5  |
| g++      | 9.3.0  |
| gcc      | 9.3.0  |

python库的环境列表：

```
Package                          Version            
-------------------------------- -------------------
absl-py                 0.13.0
astunparse              1.6.3
brotlipy                0.7.0
cachetools              4.2.2
certifi                 2020.6.20
cffi                    1.14.3
chardet                 3.0.4
click                   8.0.1
conda                   4.9.2
conda-package-handling  1.7.2
cryptography            3.2.1
filelock                3.0.12
flatbuffers             1.12
gast                    0.4.0
google-auth             1.33.1
google-auth-oauthlib    0.4.4
google-pasta            0.2.0
grpcio                  1.34.1
h5py                    3.1.0
idna                    2.10
jieba                   0.42.1
joblib                  1.0.1
keras-nightly           2.5.0.dev2021032900
Keras-Preprocessing     1.1.2
Markdown                3.3.4
numpy                   1.19.5
oauthlib                3.1.1
opt-einsum              3.3.0
packaging               21.0
Pillow                  8.3.1
pip                     21.1.3
protobuf                3.17.3
pyasn1                  0.4.8
pyasn1-modules          0.2.8
pycosat                 0.6.3
pycparser               2.20
pyOpenSSL               19.1.0
pyparsing               2.4.7
PySocks                 1.7.1
regex                   2021.7.6
requests                2.24.0
requests-oauthlib       1.3.0
rsa                     4.7.2
ruamel-yaml             0.15.87
sacremoses              0.0.45
scikit-learn            0.24.0
scipy                   1.7.0
setuptools              50.3.1.post20201107
six                     1.15.0
tensorboard             2.5.0
tensorboard-data-server 0.6.1
tensorboard-plugin-wit  1.8.0
tensorflow              2.5.0
tensorflow-estimator    2.5.0
termcolor               1.1.0
threadpoolctl           2.2.0
tokenizers              0.10.3
torch                   1.9.0+cu111
torchaudio              0.9.0
torchvision             0.10.0+cu111
tqdm                    4.51.0
transformers            4.3.3
typing-extensions       3.7.4.3
urllib3                 1.25.11
Werkzeug                2.0.1
wheel                   0.35.1
wrapt                   1.12.1
```

等待补全中

如果你有需要的环境，请在github issue提出申请，或请联系比赛管理员进行安装。

## 预训练语言模型
因评测环境不允许连接互联网，若选手需要使用预训练语言模型，则需要自己手动上传。同时，评测环境中已经支持`transformers`中部分预训练语言模型，可直接通过 `model = AutoModel.from_pretrained(modelname)` 进行加载。已支持的预训练语言模型包括：
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
