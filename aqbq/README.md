# CAIL2021 —— 案情标签预测

该项目为 **CAIL2021——案情标签预测** 的代码和模型提交说明

报名地址 [[link]](http://cail.cipsc.org.cn/task8.html?raceID=6)， 数据集下载 [[link]](http://cail.cipsc.org.cn/datagrid.html?raceID=6)，CAIL2021官网[[link]](http://cail.cipsc.org.cn/index.html)

选手交流QQ群：237633234

## 数据说明

本任务所使用的数据集来自于裁判文书网公开的裁判文书。

下发的文件包含 `train.json` 和 `tree.json`, 分别是训练集和案情标签体系文件.

**`tree.json`中所包含的案情标签体系信息如下：**

- 案情标签体系是一个树状结构，各级标签存在对应关系
- 共包含11个`一级标签`，92个 `二级标签`，252个 `三级标签`
- 其中，少部分 `二级标签` 和 `三级标签` 相同的情况是正常的

**`train.json`包含若干条数据，每条数据的字段信息如下：**

- `termID`:  代表该条数据的ID
- `id`: 另一种数据标识符
- `content`:  结构化处理之后的裁判文书的内容
- `result`: 整篇文书所对应的所有案情标签
  - 每条案情标签格式：`一级标签/二级标签/三级标签` 
  - result中包含多条案情标签
- `evidence`:  每个案情标签在文书中所对应的依据，包含以下两个字段
  - `index`：content中所对应句子的索引
  - `value`：该句子所对应的案情标签

最终测试集中不包含 `result` 和 `evidence` 字段，选手根据其他字段信息预测`result`，最终评测时**只考虑** `result` 字段，不考虑 `evidence` 字段

## 提交的文件格式及组织形式

你可以在`python_sample`中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个`zip`文件进行提交，该`zip`文件内部形式可以参看`python_sample/main.zip`。该`zip`文件**内部顶层**必须包含`main.py`，为运行的入口程序，我们会在该目录下使用`python3 main.py`来运行你的程序。

## 代码的内容

对于你的代码，你需要从`/input/`中读取数据进行预测。

在该文件夹中包含**若干**文件，每个文件均由若干行`json`格式数据组成。每行的数据格式与下发数据格式完全一致。选手需要将预测的结果输出到`/output/result.txt`中，以`json`格式输出一个列表。你需要按照测试集的顺序，依次输出每个测试样本所对应的标签，类型为`list`。（格式详见`Baseline/output/result.txt`）

以上为`main.py`中你需要实现的内容，你可以利用`python_sample`下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到`/work`路径下然后运行。

## 其他语言的支持

如上文所述，我们现阶段只支持`python`语言的提交，但是这并不代表你不能够使用其他语言进行预测。你可以使用`python3 main.py`去调用运行其他语言的命令。但请注意，在你调用其他命令的时候请在命令前加上`sudo`以保证权限不会出问题。

## 现有的系统环境

| 软件名称 | 版本号 |
| -------- | :----- |
| python   | 3.8.10 |
| g++      | 8.4.0  |
| gcc      | 8.4.0  |

python库的环境列表：

```
Package                Version
---------------------- --------------------
attrs                  19.3.0
Automat                0.8.0
backcall               0.2.0
blinker                1.4
certifi                2019.11.28
chardet                3.0.4
click                  8.0.1
cloud-init             20.3
colorama               0.4.3
command-not-found      0.3
configobj              5.0.6
constantly             15.1.0
cryptography           2.8
dbus-python            1.2.16
decorator              4.4.2
distro                 1.4.0
distro-info            0.23ubuntu1
entrypoints            0.3
Flask                  2.0.1
future                 0.18.2
gevent                 21.1.2
greenlet               1.1.0
gunicorn               20.1.0
httplib2               0.14.0
hyperlink              19.0.0
idna                   2.8
importlib-metadata     1.5.0
incremental            16.10.1
ipython                7.18.1
ipython-genutils       0.2.0
itsdangerous           2.0.1
jedi                   0.17.2
jieba                  0.42.1
Jinja2                 3.0.1
joblib                 1.0.1
jsonpatch              1.22
jsonpointer            2.0
jsonschema             3.2.0
keyring                18.0.1
language-selector      0.1
launchpadlib           1.10.13
lazr.restfulclient     0.14.2
lazr.uri               1.0.3
MarkupSafe             2.0.1
more-itertools         4.2.0
netifaces              0.10.4
numpy                  1.19.2
oauthlib               3.1.0
pandas                 1.3.0
parso                  0.7.1
pexpect                4.6.0
pickleshare            0.7.5
pip                    20.0.2
prettytable            0.7.2
prompt-toolkit         3.0.8
pyasn1                 0.4.2
pyasn1-modules         0.2.1
Pygments               2.7.1
PyGObject              3.36.0
PyHamcrest             1.9.0
PyJWT                  1.7.1
pymacaroons            0.13.0
PyNaCl                 1.3.0
pyOpenSSL              19.0.0
pyrsistent             0.15.5
pyserial               3.4
python-apt             2.0.0+ubuntu0.20.4.3
python-dateutil        2.8.2
python-debian          0.1.36ubuntu1
pytz                   2021.1
PyYAML                 5.3.1
requests               2.22.0
requests-unixsocket    0.2.0
scikit-learn           0.24.2
scipy                  1.7.0
SecretStorage          2.3.1
service-identity       18.1.0
setuptools             45.2.0
simplejson             3.16.0
six                    1.14.0
sklearn                0.0
sos                    4.0
ssh-import-id          5.10
systemd-python         234
tensorflow             2.5.0
threadpoolctl          2.2.0
torch                  1.6.0
tqdm                   4.50.2
traitlets              5.0.5
transformers           4.3.3
Twisted                18.9.0
ubuntu-advantage-tools 20.3
ufw                    0.36
unattended-upgrades    0.1
urllib3                1.25.8
wadllib                1.3.3
wcwidth                0.2.5
Werkzeug               2.0.1
wheel                  0.34.2
zipp                   1.0.0
zope.event             4.5.0
zope.interface         4.7.1
```

