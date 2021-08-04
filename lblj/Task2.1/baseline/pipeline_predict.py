# -*- encoding: utf-8 -*-

"""
用途      ：全流程预测模块
@File    :pipeline_predict.py
@Time    :2021/7/16 15:47
@Author  :hyyd
@Software: PyCharm
"""
import fire

from prepare_after_step1 import matrix_pooling_to_feature_vector
from step1_model.bert.bert_predict import create_prediction_model, single_file_predict_pipeline
from step2_model.data_util import LabelTransform
from step2_model.model_predict import MultiLabelPredictModel
from utils import *
import numpy as np


def extract_data(dataset):
    """
    从数据中剥离输入数据
    :param dataset:
    :return:
    """
    new_dataset = []
    tag_list = []
    for data in dataset:
        new_dataset.append(data[1])
        tag_list.append(data[0])
    return new_dataset, tag_list


class PipelinePredict(object):
    def __init__(self, step1_model, step2_model, label_list_path):
        self.step1_model = step1_model
        self.step2_model = step2_model
        self.label_list = txt_to_list(label_list_path)
        self.lt = LabelTransform(self.label_list)

    def step1_predict(self, data):
        _, prediction_list = single_file_predict_pipeline(data, self.step1_model)
        feature_vector = matrix_pooling_to_feature_vector(prediction_list)
        return feature_vector

    def step2_predict(self, np_list):
        result = self.step2_model.predict(np_list)
        labels = self.lt.multi_hot_to_labels(result[0])
        return labels

    def pipeline_predict(self, data):
        vector = self.step1_predict(data)
        vector = np.array(vector, dtype=np.float)
        vector = np.expand_dims(vector, axis=0)
        labels = self.step2_predict(vector)
        return labels


def run_predict(dataset, step1_model, step2_model, label_list_path):
    """
    执行整体的predict流程
    :param dataset: 原始输入，文本列表数据
    :param step1_model:
    :param step2_model:
    :param label_list_path:
    :return:
    """

    prediction_labels = []
    pipeline_predictor = PipelinePredict(step1_model, step2_model, label_list_path)
    for data in dataset:
        labels = pipeline_predictor.pipeline_predict(data)
        prediction_labels.append(labels)
    return prediction_labels


def main(label_list_path=r"./data/step2_data/labels.txt",
         test_data_path=r"./data/single_file_data/test_data.txt",
         step2_model_path=r"./step2_model/output/model.pkl",
         output_prediction_path=r"./data/output/prediction.txt",
         output_tag_path=r"./data/output/criterion.txt",
         vocab_file='./step1_model/BERT_BASE_DIR/vocab.txt',
         bert_config_file='./step1_model/BERT_BASE_DIR/bert_config.json',
         ckpt_dir='./step1_model/output',
         max_seq_len=500,
         n_classes=133):
    step1_model = create_prediction_model(vocab_file=vocab_file, bert_config_file=bert_config_file, ckpt_dir=ckpt_dir,
                                          max_seq_len=max_seq_len, n_classes=n_classes)
    step2_model = MultiLabelPredictModel(step2_model_path)
    test_dataset = txt_to_matrix(test_data_path)
    new_test_dataset, tags = extract_data(test_dataset)
    prediction = run_predict(new_test_dataset, step1_model, step2_model, label_list_path)
    matrix_to_txt(output_prediction_path, prediction)
    list_to_txt(output_tag_path, tags)


if __name__ == '__main__':
    fire.Fire(main)
