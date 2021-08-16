# CAIL2021——阅读理解

该项目为 **CAIL2021—阅读理解** 的代码和模型提交说明。

## 选手交流群

QQ群：237633234

## 任务介绍

在法律问答任务中，很多问题需要通过文章中多个片段组合出最终的答案。因此，本次中文法律阅读理解比赛引入多片段回答的问题类型，即部分问题需要抽取文章中的多个片段组合成最终答案。希望多片段问题类型的引入，能够扩大中文机器阅读理解的场景适用性。本次比赛依旧保留单片段、是否类和拒答类的问题类型。

## 数据说明

本任务技术评测的训练集包括两部分，一部分来源于[CAIL2019](https://github.com/china-ai-law-challenge/CAIL2019/tree/master/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3/data)和[CAIL2020](https://github.com/china-ai-law-challenge/CAIL2020/tree/master/ydlj/data)的训练集，一部分为重新标注的约4000个问答对的训练集。验证集和测试集分别约1500个问答对。第一阶段为小规模训练集和验证集，第二阶段为全量训练集和验证集，第三阶段使用测试集用于封测。

数据格式参考SquAD2.0的数据格式，整体为json格式的数据。并增设案由"casename"字段。

## 提交的文件格式及组织形式

你可以在 ``model`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``model/submit_sample.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``../data/data.json``中读取数据进行预测，该数据格式与下发数据格式完全一致，隐去答案信息。选手需要将预测的结果输出到``../result/result.json``中，预测结果文件为一个json格式的文件，具体可以查看 ``evaluate/result.json``。

你可以利用 ``model`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/model``路径下然后运行。

我们提供基于[legal-ELECTRA-base中文预训练模型](https://github.com/ymcui/Chinese-ELECTRA)的基线模型（训练代码），放置在``baseline``目录下，供选手参考。

并提供基线模型的[提交版本](http://pan.iflytek.com:80/link/C0EFACE49E856F51F4823DD866583C29)(有效期限：2021-08-28，访问密码：HyKE)

## 评测脚本

评价采用F1宏平均（Macro-Average F1），与CAIL2019评价方法一致，对于多片段回答类型，将多片段组合成一个答案后按照单片段方式来计算。

我们在 ``evaluate`` 文件夹中提供了评分的代码，以供参考。

## 现有的系统环境

[torch1.4+transformers3.4](https://github.com/china-ai-law-challenge/CAIL2021/edit/main/ydlj/envs/torch1.4+transformers3.4.md)
[torch1.8+transformers4.9](https://github.com/china-ai-law-challenge/CAIL2021/edit/main/ydlj/envs/torch1.8+transformers4.9.md)

另外评测使用的GPU型号是Tesla V100G32。

如果你有需要的环境，请联系比赛管理员进行安装。

## 关注我们
欢迎关注哈工大讯飞联合实验室官方微信公众号，了解最新的技术动态。

<div align=center><img width="400" height="400" src="https://github.com/china-ai-law-challenge/CAIL2021/edit/main/ydlj/images/HFL.jpg"/></div>

## 问题反馈
如有问题，请在GitHub Issue中提交
