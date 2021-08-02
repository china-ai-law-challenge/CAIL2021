# CAIL2021 —— 类案检索

该项目为 **CAIL2021——类案检索** 的代码和模型提交说明

## 任务介绍

该任务为面向中国刑事案件的类案检索。具体地，给定若干个查询案例（query），每一个查询案例各自对应一个大小为100的候选案例（candidate）池，要求从候选案例池中筛选出与查询案例相关的类案。类案相似程度划分为四级（从最相关：3 到完全不相关：0），判定标准详见[类案标注文档](https://docs.qq.com/doc/DU1FTbWZtcnpBVnhx)。每个查询案例最终的提交形式为对应的100个候选案例的排序列表，预测越相似的案例排名越靠前。

## 数据集说明

本任务所使用的数据集来自于裁判文书网公开的裁判文书。其中初赛阶段全部数据、复赛阶段训练集、封测阶段训练集均使用公开的中文类案检索数据集[LeCaRD](https://github.com/myx666/LeCaRD)。以初赛阶段数据集为例，文件结构如下：

```
input
├── candidates
│   ├── 111
│   ├── 222
│   ├── 333
│   ├── 444
│   └── 555
└── query.json

6 directories, 1 file
```

其中，input是输入文件根目录，包含了两个部分：`query.json`和`candidates/`。
`query.json`包括了该阶段所有的query，每个query均以字典格式进行存储。下面是一个query的示例：

```
{"path": "ba1a0b37-3271-487a-a00e-e16abdca7d83/005da2e9359b1d71ae503d98fba4d3f31b1.json", "ridx": 1325, "q": "2016年12月15日12时许，被害人郑某在台江区交通路工商银行自助ATM取款机上取款后，离开时忘记将遗留在ATM机中的其所有的卡号为62×××73的银行卡取走。后被告人江忠取钱时发现该卡处于已输入密码的交易状态下，遂分三笔取走卡内存款合计人民币（币种，下同）6500元。案发后，被告人江忠返还被害人郑某6500元并取得谅解。", "crime": ["诈骗罪", "信用卡诈骗罪"]}
```

query的各个字段含义如下：
- **path**：查询案例对应的判决书在原始数据集中的位置（在本次比赛中不重要，可以忽略）
- **ridx**：每个查询案例唯一的ID
- **q**：查询案例的内容（只包含案情描述部分）
- **crime**：查询案例涉及的罪名

`candidates/`下有若干个子文件夹，每个子文件夹包含了一个query的全部100个candidates；子文件夹名称对应了其所属query的**ridx**。这100个candidate分别以字典的格式单独存储在json文件中，下面是一个candidate的示例：

```
{"ajId":"dee49560-26b8-441b-81a0-6ea9696e92a8","ajName":"程某某走私、贩卖、运输、制造毒品一案","ajjbqk":" 公诉机关指控，2018年3月1日下午3时许，被告人程某某在本市东西湖区某某路某某工业园某某宾馆门口以人民币300元的价格向吸毒人员张某贩卖毒品甲基苯丙胺片剂5颗......","pjjg":" 一、被告人程某某犯贩卖毒品罪，判处有期徒刑十个月......","qw":"湖北省武汉市东西湖区人民法院 刑事判决书 （2018）鄂0112刑初298号 公诉机关武汉市东西湖区人民检察院。 被告人程某某......","writId": "0198ec7627d2c78f51e5e7e3862b6c19e42", "writName": "程某某走私、贩卖、运输、制造毒品一审刑事判决书"}
```

candidate的各个字段含义如下：
- ajId：候选案例的ID（可忽略）
- ajName：案例的名称
- ajjbqk：案件基本情况
- pjjg：法院的判决结果
- qw：判决书的全文内容
- writID：判决书的ID（可忽略）
- writName是判决书的名称（可忽略）

一些注意事项：
- 查询案例的ID（ridx）可能为正整数（例如1325）或者负整数（例如-991），但是本次比赛中并不加以区分，只需要看作唯一对应的ID序号，其数值不具有任何含义。
- 根据组委会要求，初赛阶段仅使用25%的LeCaRD数据作为训练集和测试集；复赛阶段将使用LeCaRD全集作为训练接；复赛阶段和封闭评测阶段都将使用不公开的数据进行评测，但是数据结构、数据类型与前两个阶段保持一致。
- LeCaRD数据集的论文原文。如果您在CAIL评测中或者评测后引用LeCaRD数据集发表论文或取得科研成果，请您在发表论文和申报成果时声明“使用了LeCaRD数据集”，并按如下格式引用：
```
Yixiao Ma, Yunqiu Shao, Yueyue Wu, Yiqun Liu∗, Ruizhe Zhang, Min Zhang, Shaoping Ma. 2021. LeCaRD: A Legal Case Retrieval Dataset for Chinese Law System. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’21), July 11–15, 2021, Virtual Event, Canada. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3404835.3463250
```

## 提交的文件格式

你可以在`python_sample`中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个`zip`文件进行提交，该`zip`文件内部形式可以参看`python_sample/main.zip`。该`zip`文件**内部顶层**必须包含`main.py`，为运行的入口程序，我们会在该目录下使用`python3 main.py --input INPUT_PATH --output OUTPUT_PATH`来运行你的程序。在正式测试程序时，后端脚本将自动分配合适的参数路径，因此你无需关心`--input`和`--output`参数。

## 代码的内容

对于你的代码，你需要从`args.input`中读取数据进行类案检索，`args.input`即为前面提到的`input/`文件夹路径，格式也保持一致。在完成了任务、得到每个query的candidate排序列表后，需要将结果保存到`args.input`下且**必须**命名为`prediction.json`。结果以字典格式存储，下面是一个`prediction.json`的示例（数字仅供示意）：

```
{ "111": [12, 7, 87, ... ], "222": [8765, 543, 777, ... ], "-32": [99, 342, 30, ...] ... }
```

请注意：提交结果的字典务必包括**全部**query的`ridx`作为key，并且由于本次类案检索任务的评测指标是NDCG@30，所以每个key下对应的列表长度至少为30（建议为100）。提交格式的错误将会直接影响评测结果！

## 评测指标

类案检索任务的评测指标为[NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)@30，也即结果列表前30位candidate的NDCG（Normalized Discounted Cumulative Gain）值。复赛阶段至多可提交三个模型，以最高NDCG@30分数作为复赛阶段成绩；封闭评测阶段。参赛者可自主从复赛阶段提交的三个模型中指定任意一个模型为最终模型，其在封测数据集上的NDCG@30分数计为封测阶段成绩。

## 其他语言的支持

如上文所述，我们现阶段只支持`python`语言的提交，但是这并不代表你不能够使用其他语言进行预测。你可以使用`python3 main.py --input INPUT_PATH --output OUTPUT_PATH`去调用运行其他语言的命令。

## 现有的系统环境

| 软件名称 | 版本号 |
| -------- | :----- |
| python   | 3.6.3 |
| g++      | 7.5.0  |
| gcc      | 7.5.0  |

python库的环境列表：

```
Package                            Version
---------------------------------- --------------------
absl-py                            0.13.0
alabaster                          0.7.10
anaconda-client                    1.6.5
anaconda-navigator                 1.6.9
anaconda-project                   0.8.0
appdirs                            1.4.4
asgiref                            3.3.1
asn1crypto                         0.22.0
astroid                            1.5.3
astropy                            2.0.2
Babel                              2.5.0
backports.shutil-get-terminal-size 1.0.0
beautifulsoup4                     4.6.0
bitarray                           0.8.1
bkcharts                           0.2
blaze                              0.11.3
bleach                             2.0.0
bokeh                              0.12.10
boto                               2.48.0
boto3                              1.15.7
botocore                           1.18.7
Bottleneck                         1.2.1
cachetools                         4.2.2
certifi                            2021.5.30
cffi                               1.10.0
chardet                            3.0.4
charset-normalizer                 2.0.3
click                              6.7
cloudpickle                        0.4.0
clyent                             1.2.2
colorama                           0.3.9
conda                              4.9.2
conda-build                        3.0.27
conda-package-handling             1.7.2
conda-verify                       2.0.0
contextlib2                        0.5.5
cryptography                       2.0.3
cycler                             0.10.0
Cython                             0.26.1
cytoolz                            0.8.2
dask                               0.15.3
dataclasses                        0.8
datashape                          0.5.4
decorator                          4.1.2
distlib                            0.3.2
distributed                        1.19.1
Django                             1.10.8
django-haystack                    2.5.1
docker                             4.4.0
docutils                           0.14
english                            2020.7.0
entrypoints                        0.2.3
et-xmlfile                         1.0.1
fastcache                          1.0.2
filelock                           3.0.12
Flask                              0.12.2
Flask-Cors                         3.0.3
future                             0.18.2
gensim                             3.8.3
gevent                             1.2.2
glob2                              0.5
gmpy2                              2.0.8
google-auth                        1.34.0
google-auth-oauthlib               0.4.5
greenlet                           0.4.12
grpcio                             1.39.0
h5py                               2.7.0
heapdict                           1.0.0
html5lib                           0.999999999
idna                               3.2
imageio                            2.2.0
imagesize                          0.7.1
importlib-metadata                 4.6.1
importlib-resources                5.2.0
ipykernel                          4.6.1
ipython                            6.1.0
ipython-genutils                   0.2.0
ipywidgets                         7.0.0
isort                              4.2.15
itsdangerous                       0.24
jdcal                              1.3
jedi                               0.10.2
jieba                              0.42.1
Jinja2                             2.9.6
jmespath                           0.10.0
joblib                             0.17.0
jsonschema                         2.6.0
jupyter-client                     5.1.0
jupyter-console                    5.2.0
jupyter-core                       4.3.0
jupyterlab                         0.27.0
jupyterlab-launcher                0.4.0
langdetect                         1.0.8
langid                             1.1.6
lazy-object-proxy                  1.3.1
lightgbm                           3.1.0
llvmlite                           0.20.0
locket                             0.2.0
lxml                               4.1.0
Markdown                           3.3.4
MarkupSafe                         1.0
matplotlib                         2.1.0
mccabe                             0.6.1
mistune                            0.7.4
mkl-fft                            1.0.6
mkl-random                         1.0.1
monotonic                          1.5
mpmath                             0.19
msgpack-python                     0.4.8
multipledispatch                   0.4.9
navigator-updater                  0.1.0
nbconvert                          5.3.1
nbformat                           4.4.0
neotime                            1.7.4
networkx                           2.0
nltk                               3.2.4
nose                               1.3.7
notebook                           5.0.0
numba                              0.35.0+10.g143f70e90
numexpr                            2.6.2
numpy                              1.19.5
numpydoc                           0.7.0
oauthlib                           3.1.1
odo                                0.5.1
olefile                            0.44
openpyxl                           2.4.8
packaging                          16.8
pandas                             1.1.5
pandocfilters                      1.4.2
pansi                              2020.7.3
partd                              0.3.8
path.py                            10.3.1
pathlib2                           2.3.0
patsy                              0.4.1
pep8                               1.7.0
pexpect                            4.2.1
pickleshare                        0.7.4
Pillow                             4.2.1
pip                                21.1.3
pkginfo                            1.4.1
ply                                3.10
prettytable                        2.0.0
prompt-toolkit                     2.0.10
protobuf                           3.17.3
psutil                             5.4.0
ptyprocess                         0.5.2
py                                 1.4.34
py2neo                             2020.1.1
pyasn1                             0.4.8
pyasn1-modules                     0.2.8
pycodestyle                        2.3.1
pycosat                            0.6.3
pycparser                          2.18
pycrypto                           2.6.1
pycurl                             7.43.0
pyecharts                          1.9.0
pyflakes                           1.6.0
Pygments                           2.2.0
pylint                             1.7.4
PyMySQL                            0.10.1
pyodbc                             4.0.17
pyOpenSSL                          17.2.0
pyparsing                          2.2.0
PySocks                            1.6.7
pytest                             3.2.1
python-dateutil                    2.8.2
pytorch-pretrained-bert            0.6.2
pytz                               2021.1
PyWavelets                         0.5.2
PyYAML                             3.12
pyzmq                              16.0.2
QtAwesome                          0.4.4
qtconsole                          4.3.1
QtPy                               1.3.1
rank-bm25                          0.2.1
regex                              2020.9.27
requests                           2.26.0
requests-oauthlib                  1.3.0
rope                               0.10.5
rsa                                4.7.2
ruamel-yaml                        0.11.14
s3transfer                         0.3.3
sacremoses                         0.0.45
scikit-image                       0.13.0
scikit-learn                       0.23.2
scipy                              0.19.1
seaborn                            0.8
setuptools                         57.4.0
simplegeneric                      0.8.1
simplejson                         3.17.2
singledispatch                     3.4.0.3
six                                1.16.0
sklearn                            0.0
smart-open                         4.1.2
snowballstemmer                    1.2.1
sortedcollections                  0.5.3
sortedcontainers                   1.5.7
Sphinx                             1.6.3
sphinxcontrib-websupport           1.0.1
spyder                             3.2.4
SQLAlchemy                         1.1.13
sqlparse                           0.4.1
statsmodels                        0.8.0
sympy                              1.1.1
tables                             3.4.2
TBB                                0.1
tblib                              1.3.2
tensorboard                        2.5.0
tensorboard-data-server            0.6.1
tensorboard-plugin-wit             1.8.0
tensorboardX                       1.8
terminado                          0.6
testpath                           0.3.1
threadpoolctl                      2.1.0
thulac                             0.2.1
tokenizers                         0.9.4
toolz                              0.8.2
torch                              1.2.0
torch-tb-profiler                  0.2.1
torchvision                        0.7.0
tornado                            4.5.2
tqdm                               4.54.0
traitlets                          4.3.2
transformers                       4.0.0
typing                             3.6.2
typing-extensions                  3.10.0.0
tzlocal                            2.0.0
unicodecsv                         0.14.1
urllib3                            1.26.6
virtualenv                         20.4.7
wcwidth                            0.1.7
webencodings                       0.5.1
websocket-client                   0.57.0
Werkzeug                           2.0.1
wheel                              0.36.2
Whoosh                             2.7.4
widgetsnbextension                 3.0.2
wrapt                              1.10.11
xlrd                               1.1.0
XlsxWriter                         1.0.2
xlwt                               1.3.0
zict                               0.1.3
zipp                               3.5.0
```

如果你有其他环境或者python库要求，可以在issue中写明具体需求。