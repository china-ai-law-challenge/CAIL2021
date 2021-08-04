# -*- encoding: utf-8 -*-

"""
用途      ：
@File    :model_train.py
@Time    :2021/7/16 13:42
@Author  :hyyd
@Software: PyCharm
"""
import sys

sys.path.append('..')

from sklearn.ensemble import RandomForestClassifier

import numpy as np
from step2_model.data_util import DataLoader, LabelTransform
from pipeline_evaluate import cal_score

from utils import *

import fire


def train(train_dataset, label_list):
    # 数据处理和转化
    data_loader = DataLoader(train_dataset, label_list)
    X_data = data_loader.get_data_x()
    X_data = np.array(X_data, dtype=np.float)
    Y_data = data_loader.get_data_y()
    Y_data = np.array(Y_data, dtype=np.int)

    # 模型构建
    clf = RandomForestClassifier()
    clf.fit(X=X_data, y=Y_data)
    return clf


def evaluate(model, test_dataset, label_list):
    labels = [data[0] for data in test_dataset]
    data_loader = DataLoader(test_dataset, label_list)
    lt = LabelTransform(label_list)
    X_data = data_loader.get_data_x()
    X_data = np.array(X_data, dtype=np.float)
    predictions = model.predict(X_data)
    preds = []
    for prediction in predictions:
        pred = lt.multi_hot_to_labels(prediction)
        preds.append(pred)
    res = cal_score(preds, labels)
    return res


def main(train_data_path=r"../data/step2_data/train_data.txt",
         test_data_path=r"../data/step2_data/test_data.txt",
         label_list_path=r"../data/step2_data/labels.txt",
         model_save_path=r"./output/model.pkl" ):
    train_dataset = txt_to_matrix(train_data_path)
    test_dataset = txt_to_matrix(test_data_path)
    label_list = txt_to_list(label_list_path)

    clf = train(train_dataset, label_list)

    score = evaluate(clf, test_dataset, label_list)

    write_pickle(model_save_path, clf)



if __name__ == '__main__':
    fire.Fire(main)
