# CAIL2021——司法文本信息抽取

该项目为 **CAIL2021—司法文本信息抽取** 的代码和模型提交说明。

## 选手交流群

QQ群：237633234

## 数据说明

本次任务所使用的数据集主要来自于网络公开的若干罪名起诉意见书，总计7500余条样本，10类相关业务相关实体，分别为犯罪嫌疑人、受害人、作案工具、被盗物品、被盗货币、物品价值、盗窃获利、时间、地点、组织机构。其中第一阶段数据集包含约1700条样本。每条样本中均包含任意数目的实体。考虑到多类罪名案件交叉的复杂性，本次任务仅涉及盗窃罪名的相关信息抽取。

针对本次任务，我们会提供包含案件情节描述的陈述文本，选手需要识别出文本中的关键信息实体，并按照规定格式返回结果。

发放的文件为``xxcq_small.json``，为字典列表，字典包含字段为：

- ``id``：案例中句子的唯一标识符。
- ``context``：句子内容，抽取自起诉意见书的事实描述部分。
- ``entities``：句子所包含的实体列表。
- ``label``：实体标签名称。
- ``span``：实体在``context``中的起止位置。

其中``label``的十种实体类型分别为：

|label|含义|
|---|---|
|NHCS|犯罪嫌疑人|
|NHVI|受害人|
|NCSM|被盗货币|
|NCGV|物品价值|
|NCSP|盗窃获利|
|NASI|被盗物品|
|NATS|作案工具|
|NT|时间|
|NS|地点|
|NO|组织机构|


## 提交的文件格式及组织形式

你可以在 ``model`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``model/submit_sample.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``/input/input.json``中读取数据进行预测，该数据格式与下发数据格式完全一致，隐去``entities``字段信息。选手需要将预测的结果输出到``/output/output.json``中，预测结果文件为一个json格式的文件，包含两个字段，分别为``id``和``entities``，具体可以查看 ``evaluate/result.json``。

你可以利用 ``model`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/model``路径下然后运行。


## 评测脚本

我们在 ``evaluate`` 文件夹中提供了评分的代码和提交文件样例，以供参考。

## 现有的系统环境

[tf2&pytorch](./envs/tf2.md)

[tf1&pytorch](./envs/tf1.md)

## 问题反馈
如有问题，请在GitHub Issue中提交