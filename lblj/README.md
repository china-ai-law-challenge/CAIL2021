# CAIL2021-论辩理解
本项目为 **中国法研杯司法人工智能挑战赛（CAIL2021）**-- **论辩理解**赛道的项目说明。

本赛道面向裁判文书中诉辩双方的表述文本进行论辩分析，旨在自动化识别双方表述中的争议观点对，形成争议焦点，并对争议焦点进行类型识别。
赛道分成两个子任务，包括争议观点对抽取以及争议点类型识别。

### 1. 争议观点对抽取

该任务旨在自动化抽取出裁判文书中诉辩双方观点陈述中存在互动关系的论点对。
任务设置如下：给定一个诉方观点和五个候选观点，模型需要自动识别出候选观点中哪一个是能形成争议的辩方观点。
本任务有两个设定：

  #### 1.1 案件类型不敏感的争议观点对抽取
  
  #### 1.2 跨案件类型的争议观点对迁移学习

### 2. 争议点类型识别

该任务的目标是自动识别裁判文书中的争议焦点类型。
任务设置如下：给定裁判文书诉请、抗辩和裁判分析过程等三个段落，根据段落中提供的信息判断文书是否存在争议焦点，争议焦点是哪种类型。
 
