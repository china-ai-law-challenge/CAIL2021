# 1.2 跨案件类型的争议观点对迁移学习（CAIL2021--论辩理解）

该项目为 **CAIL2021—论辩理解** 赛道任务1.1 **跨案件类型的争议观点对迁移学习** 的代码和模型提交说明。

数据集下载请访问比赛[主页](http://cail.cipsc.org.cn/)。


## 选手交流群

QQ群：

## 任务说明

本任务提供的训练数据和测试数据来源于两个不同的领域的裁判文书：训练数据来自刑事案件（采用与任务1.1完全一致的数据）、测试数据来自海事海商案件。本任务需要参赛者设计的算法能在仅获取单一领域的数据标注的情况下，在跨领域的测试集上有良好的表现。

## 数据说明

本任务第一阶段所下发的文件包含``SMP-CAIL2021-text-train.csv``,`` SMP-CAIL2021-train.csv``,``SMP-CAIL2021-crossdomain-text-test1.csv``,``SMP-CAIL2021-crossdomain-test1.csv``，分别包含以下内容：

1. ``SMP-CAIL2021-text-train.csv``：包含了裁判文书所有对于辩诉双方辩护全文的数据。分别包含以下维度：
   	<br/>``sentence_id``： 句子id
      	<br/>``text_id``： 裁判文书id
      	<br/>``position``： 二分类标签：sc——诉方；bc——辩方
      	<br/>``sentence``： 句子文本

2. ``SMP-CAIL2021-train.csv``： 包含了2449对裁判文书中的互动论点对。分别包含以下维度：
    <br/>``id``： 论点对id
    <br/>``text_id``： 裁判文书id
    <br/>``sc``： 论点对中诉方论点
    <br/>``A/B/C/D/E``： 给出的五句候选辩方论点
    <br/>``answer``： 辩方正确论点

3. ``SMP-CAIL2021-crossdomain-text-test1.csv``：同下发数据中的``SMP-CAIL2021-text-train.csv``格式完全一致；
4. ``SMP-CAIL2021-crossdomain-test1.csv``：同下发数据中的``SMP-CAIL2021-train.csv``格式基本一致，包含了302对裁判文书中的互动论点对，但缺少选手``answer``维度数据。

## 提交的文件格式及组织形式

你可以在 ``python_sample`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``python_sample/main.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``/input/``中读取数据进行预测。

第一阶段测试中，该文件夹包含**两个**文件：``SMP-CAIL2021-crossdomain-text-test1.csv``和``SMP-CAIL2021-crossdomain-test1.csv``。选手需要从将预测的结果输出到``/output/result1.csv``中，以``csv``格式输出。输出文件中必须包含且只包含``id``和``answer``两个维度，可参考下发文件夹中``Sample_submission.csv``。

以上为 ``main.py`` 中你需要实现的内容，你可以利用 ``python_example`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/work``路径下然后运行。

## 其他语言的支持

如上文所述，我们现阶段只支持 ``python`` 语言的提交，但是这并不代表你不能够使用其他语言进行预测。你可以使用``python3 main.py``去调用运行其他语言的命令。但请注意，在你调用其他命令的时候请在命令前加上``sudo``以保证权限不会出问题。

## 基线模型

本次测评共提供了基于BERT的神经网络基线模型（选手需自行训练），放置在``baseline``目录下，供选手参考。

## 现有的系统环境

| 软件名称 | 版本号 |
| -------- | :----- |
| python   | 3.6.9  |
| g++      | 5.4.0  |
| gcc      | 5.4.0  |

python库的环境列表：

```
Package                Version
---------------------- ---------------
asn1crypto             0.24.0
backcall               0.1.0
beautifulsoup4         4.8.0
certifi                2021.5.30
cffi                   1.12.3
chardet                4.0.0
click                  8.0.1
conda                  4.7.11
conda-build            3.18.9
conda-package-handling 1.3.11
cryptography           2.7
cycler                 0.10.0
dataclasses            0.8
decorator              4.4.0
filelock               3.0.12
fire                   0.4.0
glob2                  0.7
huggingface-hub        0.0.12
idna                   2.10
importlib-metadata     4.6.1
ipython                7.8.0
ipython-genutils       0.2.0
jedi                   0.15.1
jieba                  0.42.1
Jinja2                 2.10.1
joblib                 1.0.1
kiwisolver             1.3.1
libarchive-c           2.8
lief                   0.9.0
MarkupSafe             1.1.1
matplotlib             3.3.4
mkl-fft                1.0.14
mkl-random             1.0.2
mkl-service            2.3.0
networkx               2.5.1
nltk                   3.6.2
numpy                  1.19.5
olefile                0.46
packaging              21.0
pandas                 1.1.5
parso                  0.5.1
pexpect                4.7.0
pickleshare            0.7.5
Pillow                 8.3.1
pip                    21.1.3
pkginfo                1.5.0.1
prompt-toolkit         2.0.9
psutil                 5.6.3
ptyprocess             0.6.0
pycosat                0.6.3
pycparser              2.19
Pygments               2.4.2
pyOpenSSL              19.0.0
pyparsing              2.4.7
PySocks                1.7.0
python-dateutil        2.8.1
pytz                   2021.1
PyYAML                 5.1.2
regex                  2021.7.6
requests               2.25.1
ruamel-yaml            0.15.46
sacremoses             0.0.45
scikit-learn           0.24.2
scipy                  1.5.4
setuptools             41.0.1
six                    1.16.0
sklearn                0.0
soupsieve              1.9.2
termcolor              1.1.0
threadpoolctl          2.2.0
tokenizers             0.10.3
torch                  1.7.1
torchvision            0.4.0a0+6b959ee
tqdm                   4.61.2
traitlets              4.3.2
transformers           4.8.2
typing-extensions      3.10.0.0
urllib3                1.26.6
wcwidth                0.1.7
wheel                  0.36.2
zipp                   3.5.0
```

等待补全中

如果你有需要的环境，请联系比赛管理员进行安装。
