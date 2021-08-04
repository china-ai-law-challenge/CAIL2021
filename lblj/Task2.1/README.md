# 2.1 争议点类型识别（CAIL2021--论辩理解）


该项目为 **CAIL2021—论辩理解** 赛道任务2.1 **争议点类型识别** 的代码和模型提交说明。

数据集下载请访问比赛[主页](http://cail.cipsc.org.cn/)。


## 选手交流群

QQ群：237633234

## 任务说明

本任务最终提供的训练数据和测试数据来源于多个不同的案由的裁判文书。本任务的目标是自动识别裁判文书中的争议焦点类型。任务设置如下：给定裁判文书诉请、抗辩和裁判分析过程等三个段落，根据段落中提供的信息判断文书是否存在争议焦点，争议焦点是哪种类型。

## 数据说明

本任务第一阶段所下发的文件包含``SMP-CAIL2021-focus_recognition-train.json``,``SMP-CAIL2021-focus_recognition-test1.txt``，分别包含以下内容：

1. ``SMP-CAIL2021-focus_recognition-train.json``：包含了111条数据，涉及裁判文书中的诉称段、辩称段、裁判分析过程段以及文书所对应的争议焦点、诉讼请求、抗辩事由和争议焦点子节点。具体包含以下维度：
   	<br/>``文书ID``： 文书id
      	<br/>``诉称段``： 裁判文书诉称段信息
      	<br/>``辩称段``： 裁判文书辩称段信息
      	<br/>``裁判分析过程段``： 裁判文书裁判分析过程段信息
      	<br/>``争议焦点``： 裁判文书争议焦点信息
      	<br/>``诉讼请求``： 裁判文书诉讼请求信息
      	    <br/>``文书段落``： 当前标注信息所处的段落
            <br/>``要素名称``： 当前标注信息的要素名称——标签名称
            <br/>``oValue``： 当前标注信息所涉及的最小文本范围（以字词为最小粒度）
            <br/>``sentence``： 当前标注信息所涉及的最小句子集合（以句子为最小粒度）
      	<br/>``抗辩事由``： 裁判文书抗辩事由信息
            <br/>``文书段落``： 当前标注信息所处的段落
            <br/>``要素名称``： 当前标注信息的要素名称——标签名称
            <br/>``oValue``： 当前标注信息所涉及的最小文本范围（以字词为最小粒度）
            <br/>``sentence``： 当前标注信息所涉及的最小句子集合（以句子为最小粒度）
      	<br/>``争议焦点子节点``： 裁判文书争议焦点子节点信息
            <br/>``文书段落``： 当前标注信息所处的段落
            <br/>``要素名称``： 当前标注信息的要素名称——标签名称
            <br/>``oValue``： 当前标注信息所涉及的最小文本范围（以字词为最小粒度）
            <br/>``sentence``： 当前标注信息所涉及的最小句子集合（以句子为最小粒度）

2. ``SMP-CAIL2021-focus_recognition-test1.txt``： 包含了60条数据，文件中的每一行均为诉请段、抗辩段和裁判分析过程段合并到一起的裁判文书内容。
选手需要根据每行内容预测当前行对应裁判文书的争议焦点集合。

## 提交的文件格式及组织形式

你可以在 ``python_sample`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``python_sample/main.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``/input/``中读取数据进行预测。

第一阶段测试中，该文件夹包含**一个**文件：``SMP-CAIL2021-focus_recognition-test1.txt``。选手需要从将预测的结果输出到``/output/result1.txt``中，以``txt``格式输出。输出文件中必须包含且只包含**预测争议焦点**的信息，可以是空行，表示当前文档无争议焦点，可以是一个争议焦点，可以是多个争议焦点。
**注意**当预测结果为多个争议焦点时，各个争议焦点之间需要用`' '`(单个空格)进行分割，如果当前预测结果超过3个争议焦点，则系统默认顺序取前三个争议焦点为最终的预测结果，超出部分一律舍弃不计入得分，可参考下发文件夹中``Sample_submission.txt``。

以上为 ``main.py`` 中你需要实现的内容，你可以利用 ``python_example`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/work``路径下然后运行。

## 其他语言的支持

如上文所述，我们现阶段只支持 ``python`` 语言的提交，但是这并不代表你不能够使用其他语言进行预测。你可以使用``python3 main.py``去调用运行其他语言的命令。但请注意，在你调用其他命令的时候请在命令前加上``sudo``以保证权限不会出问题。

## 基线模型

本次测评共提供了基于BERT多标签分类模型和随机森林多标签分类模型的基线系统（选手需自行训练），放置在``baseline``目录下，供选手参考。

## 现有的系统环境

| 软件名称 | 版本号 |
| -------- | :----- |
| python   | 3.8.5  |
| g++      | 5.4.0  |
| gcc      | 5.4.0  |

python库的环境列表：

```
Package                Version
---------------------- -------------------
backcall               0.2.0
beautifulsoup4         4.9.3
brotlipy               0.7.0
certifi                2020.12.5
cffi                   1.14.3
chardet                3.0.4
click                  8.0.1
conda                  4.9.2
conda-build            3.21.4
conda-package-handling 1.7.2
cryptography           3.2.1
decorator              4.4.2
dnspython              2.1.0
filelock               3.0.12
fire                   0.4.0
glob2                  0.7
huggingface-hub        0.0.12
idna                   2.10
ipython                7.19.0
ipython-genutils       0.2.0
jedi                   0.17.2
jieba                  0.42.1
Jinja2                 2.11.2
joblib                 1.0.1
libarchive-c           2.9
MarkupSafe             1.1.1
mkl-fft                1.2.0
mkl-random             1.1.1
mkl-service            2.3.0
nltk                   3.6.2
numpy                  1.19.2
olefile                0.46
packaging              21.0
pandas                 1.3.0
parso                  0.7.0
pexpect                4.8.0
pickleshare            0.7.5
Pillow                 8.1.0
pip                    20.2.4
pkginfo                1.7.0
prompt-toolkit         3.0.8
psutil                 5.7.2
ptyprocess             0.7.0
pycosat                0.6.3
pycparser              2.20
Pygments               2.7.4
pyOpenSSL              19.1.0
pyparsing              2.4.7
PySocks                1.7.1
python-dateutil        2.8.2
python-etcd            0.4.5
pytz                   2020.5
PyYAML                 5.3.1
regex                  2021.7.6
requests               2.24.0
ruamel-yaml            0.15.87
sacremoses             0.0.45
scikit-learn           0.24.2
scipy                  1.5.4
setuptools             50.3.1.post20201107
six                    1.16.0
sklearn                0.0
soupsieve              2.1
termcolor              1.1.0
threadpoolctl          2.2.0
tokenizers             0.10.3
torch                  1.7.1
torchelastic           0.2.1
torchvision            0.8.2
tqdm                   4.51.0
traitlets              5.0.5
transformers           4.8.2
typing-extensions      3.7.4.3
urllib3                1.25.11
wcwidth                0.2.5
wheel                  0.35.1
zipp                   3.5.0
```

等待补全中

如果你有需要的环境，请联系比赛管理员进行安装。
