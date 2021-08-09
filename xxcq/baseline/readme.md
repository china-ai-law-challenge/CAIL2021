## 运行
1. `python3 main.py`

## 模型
1. bert + softmax

## 数据集
1. 数据集目录为`./datasets/CAIL司法文本信息抽取`,训练时将整理好的json数据集放在该目录下。
2. 若要提交评测，数据请从`/input/input.json`复制并关闭对训练集和验证集的处理。

## 标注方式
1. baseline模型默认的标注方式为BIO标注。
2. baseline模型提供了json与BIO标注文本txt之间转换函数。注意，在json缺乏标签时，baseline将默认当前的标注为O。

## 训练
1. 你必须在`main.py`的(`train_json`,`dev_json`,`test_json`)中指定你的训练集、验证集和测试集。
2. 该baseline中默认使用训练集作为验证集。
3. 我们在baseline模型中提供了基于标签分类的训练评测。注意，此处的评测与最后提交时的评测方式并不一致。
4. 标签分类评测需要在数据集中提供分词对应的标签分类。注意，我们公布的测试集中并不包含标签。

## 测试
1. 封闭评测时，需在`main.sh`中关闭`--do_train`,`--do_eval`,并将`MODEL_NAME_OR_PATH`指向`output/best_checkpoint`(此时你应该已经将训练好的bert模型保存在best_checkpoint中)。
2. 需要提交的文件,保存为`output/test_predictions.json`,若提交至评测系统请将该文件复制到`/output/output.json`中。