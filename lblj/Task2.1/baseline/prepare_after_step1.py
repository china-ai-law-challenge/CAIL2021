# -*- encoding: utf-8 -*-

"""
用途      ：在第一步模型训练结束后，利用第一步模型进行第二步训练数据的准备
@File    :prepare_after_step1.py
@Time    :2021/7/16 10:53
@Author  :hyyd
@Software: PyCharm
"""
import numpy as np

from utils import *
from step1_model.bert.bert_predict import single_file_predict_pipeline, create_prediction_model
import fire


def matrix_pooling_to_feature_vector(matrix):
    """
    预测结果转为multi_hot数据
    :param matrix:
    :return:
    """
    score_matrix = np.array(matrix)
    feature_vector = np.max(score_matrix, axis=0)
    feature_vector[feature_vector >= 0.5] = 1
    feature_vector[feature_vector < 0.5] = 0
    return feature_vector.tolist()


def single_file_data_convert_to_step2_data(dataset, bert_model):
    step2_dataset = []
    for data in dataset:
        label, text = data
        _, prediction_list = single_file_predict_pipeline(text, bert_model)
        feature_vector = matrix_pooling_to_feature_vector(prediction_list)
        step2_dataset.append([label] + feature_vector)
    return step2_dataset


def create_step2_label_list(train_dataset, test_dataset):
    """
    利用single_file_dataset生成labels
    :param train_dataset:
    :param test_dataset:
    :return:
    """
    label_list = []
    for data in train_dataset + test_dataset:
        label = data[0]
        if label not in label_list:
            label_list.append(label)
    return label_list


def main(train_dataset_path=r"data/single_file_data/train_data.txt",
         test_dataset_path=r"data/single_file_data/test_data.txt",
         step2_train_dataset_path=r"data/step2_data/train_data.txt",
         step2_test_dataset_path=r"data/step2_data/test_data.txt",
         label_list_path=r"data/step2_data/labels.txt",
         vocab_file='./step1_model/BERT_BASE_DIR/vocab.txt',
         bert_config_file='./step1_model/BERT_BASE_DIR/bert_config.json',
         ckpt_dir='./step1_model/output',
         max_seq_len=500,
         n_classes=133):
    bert_model = create_prediction_model(vocab_file=vocab_file, bert_config_file=bert_config_file, ckpt_dir=ckpt_dir,
                                         max_seq_len=max_seq_len, n_classes=n_classes)

    train_dataset = txt_to_matrix(train_dataset_path)
    test_dataset = txt_to_matrix(test_dataset_path)

    step2_train_dataset = single_file_data_convert_to_step2_data(train_dataset, bert_model)
    step2_test_dataset = single_file_data_convert_to_step2_data(test_dataset, bert_model)
    label_list = create_step2_label_list(train_dataset, test_dataset)

    matrix_to_txt(step2_train_dataset_path, step2_train_dataset)
    matrix_to_txt(step2_test_dataset_path, step2_test_dataset)
    list_to_txt(label_list_path, label_list)


if __name__ == '__main__':
    fire.Fire(main)
