# 1.1 案件类型不敏感的争议观点对抽取（CAIL2021--论辩理解）

该项目为 **CAIL2021—论辩理解** 赛道任务1.1 **案件类型不敏感的争议观点对抽取** 的代码和模型提交说明。

数据集下载请访问比赛[主页](http://cail.cipsc.org.cn/)。


## 选手交流群

QQ群：237633234

## 论辩理解及其研究方向

论辩理解(Argumentation Understanding)是自然语言处理领域一个新颖的研究方向，与本次测评相关的四份研究工作可参考：

[Overview of SMP-CAIL2020-Argmine: The Interactive Argument-Pair Extraction in Judgement Document Challenge](https://watermark.silverchair.com/dint_a_00094.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAtswggLXBgkqhkiG9w0BBwagggLIMIICxAIBADCCAr0GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMGKJiqN9vQ5h6GIeaAgEQgIICjna_kMfhshbcKPYUAc7SKulVIRRBz_12eq_EeLzHDebNmRieMSnMEy-g4yu0jsNeLDDk56gxaq3k2kqXW2rizas3ZbQydGMkh5EVeT5iMmc-AdwUWfgjWM-JcudcLgQWMgMf2EjINDYqCys0KyCWHGSOY1aKqTnEG7FSER6NC6K6KgJvYHnhXiSl-IIcNCJMVun4lf5l8MfH3A7VwE3h-hU3xi9TAvQEMcCENdc8pj1gG0V_bJyuy4yw5zskhSrsXb79dJcfHyC182lVDgMqiLAGUI6mQokc9h_QpbM93bKlSQNOrcCGmC0NnTLWghWdJ3IgPjqyqPH---_bhtKGGyCNXM3TchF2g5zJO__YC-VxdLUoBllYMrnwnEbkZwfulKNX0KfN2pntHVSSERbnUUzEOmsa_vZdu_5qO5LVDBA2G5rDq_Y8iRLQ3cKLgIuQVvv3IBtm8qOb6_ILNzjX_oX2evfMrhfxAPfMQovPePbO2RecVDptM0FtdNWt0KuvHiDZ7OZ594XIyRu_mYWvz_0nBmOCVcyOEHu9yWIfZM8RL8uJ_v8qQI6QXxJjKXqMQZYcoOTqEigJmvuZMUT_02nWfD-VEURpew29IfO2EhmgHBqZHCdjQDU1p25WLccuiEGulEHUrhQqHPg3Z9Mk0e6f6sp_ApyggXx0oi_HF8OJRNFCfKxTJyA559ScyY96BqvPVZUrqmJmBEnMtUISmO8ezXhTp768Ctd_2s8oVmeOyv5pKHqffrn2lvnj7YHhxlXVIbn5CK3XbOdRK-hsPSty_K-2ZDqQzYlC2XCKhOvjy6UX8ZZfeI0KXxAyDlgusaLl8YEAp18XtJpLbBKbUTuWHNGSeJfQGH9HHAutbQ)
<br/>[Leveraging Argumentation Knowledge Graph for Interactive Argument Pair Identification](http://fudan-disc.com/resource/public/publication/43/Leveraging_Dialogical_Argumentation_Graph_Knowledge_in_Interactive_Argument_Pair_Identification%20(2).pdf)
<br/>[Incorporating Argument-Level Interactions for Persuasion Comments Evaluation using Co-attention Model](https://www.aclweb.org/anthology/C18-1314.pdf)
<br/>[Discrete Argument Representation Learning for Interactive Argument Pair Identification](https://arxiv.org/pdf/1911.01621)

## 数据说明

本任务第一阶段所下发的文件包含``SMP-CAIL2021-text-train.csv``,`` SMP-CAIL2021-train.csv``,``SMP-CAIL2021-text-test1.csv``,``SMP-CAIL2021-test1.csv``，分别包含以下内容：

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

3. ``SMP-CAIL2021-text-test1.csv``：同下发数据中的``SMP-CAIL2021-text-train.csv``格式完全一致；
4. ``SMP-CAIL2021-test1.csv``：同下发数据中的``SMP-CAIL2021-train.csv``格式基本一致，包含了815对裁判文书中的互动论点对，但缺少选手``answer``维度数据。

## 提交的文件格式及组织形式

你可以在 ``python_sample`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``python_sample/main.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``/input/``中读取数据进行预测。

第一阶段测试中，该文件夹包含**两个**文件：``SMP-CAIL2021-text-test1.csv``和``SMP-CAIL2021-test1.csv``。选手需要从将预测的结果输出到``/output/result1.csv``中，以``csv``格式输出。输出文件中必须包含且只包含``id``和``answer``两个维度，可参考下发文件夹中``Sample_submission.csv``。

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
threadpoolctl          2.2.0               
torch                  1.6.0               
tqdm                   4.50.2              
traitlets              5.0.5               
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

等待补全中

如果你有需要的环境，请联系比赛管理员进行安装。
