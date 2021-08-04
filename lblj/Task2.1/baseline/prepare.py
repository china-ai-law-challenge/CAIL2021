# -*- encoding: utf-8 -*-

"""
@purpose :数据预处理
@File    :prepare.py
@Time    :2021/7/15 20:46
@Author  :hyyd
@Software: PyCharm
"""
import random
from collections import defaultdict
import fire
from utils import *


def raw_data_divide(dataset):
    """
    原始数据划分训练集和测试集
    :param dataset:
    :return:
    """

    partition_dict = defaultdict(list)
    for data in dataset:
        focus_data = data['争议焦点']
        partition_dict[focus_data[0]].append(data)

    train_dataset, test_dataset = [], []
    for focus, data in partition_dict.items():
        random.shuffle(data)
        train_data = data[1:]
        train_dataset.extend(train_data)
        test_data = data[:1]
        test_dataset.extend(test_data)

    return train_dataset, test_dataset


def raw_data_convert_to_single_file_data(dataset):
    """
    原始数据集转化为单文书数据集
    :param dataset:
    :return:
    """
    tar_dataset = []
    paragraph_name_list = ['诉称段', '辩称段', '裁判分析过程段']
    for data in dataset:
        content = ""
        for paragraph_name in paragraph_name_list:
            if paragraph_name in data:
                content += remove_blank(data[paragraph_name])
        focus_label = data['争议焦点'][0]
        tar_dataset.append([focus_label, content])
    return tar_dataset


class DataSetSegment(object):
    """
    分段标注数据集
    """

    def __init__(self, text):
        self.text = remove_blank(text)
        self.label_data = dict()

    def add_label_data(self, sent, label):
        self.label_data[sent] = label


def dataset_split_to_segment_dataset(data_instance):
    segments_name_list = ['诉称段', '辩称段', '裁判分析过程段']

    segments_dataset = []

    ask_list = data_instance['诉讼请求']
    answer_list = data_instance['抗辩事由']
    child_list = data_instance['争议焦点子节点']

    for segments_name in segments_name_list:
        if segments_name not in data_instance:
            continue
        text = data_instance[segments_name]
        data_seg = DataSetSegment(text)
        for ask_data in ask_list:
            if ask_data['文书段落'] == segments_name:
                data_seg.add_label_data(ask_data['oValue'], ask_data['要素名称'])

        for answer_data in answer_list:
            if answer_data['文书段落'] == segments_name:
                data_seg.add_label_data(answer_data['oValue'], answer_data['要素名称'])

        for child_data in child_list:
            if child_data['文书段落'] == segments_name:
                data_seg.add_label_data(child_data['oValue'], child_data['要素名称'])
        segments_dataset.append(data_seg)
    return segments_dataset


def segment_data_convert_to_matrix_data(seg_data):
    """
    分段数据转换为二维矩阵的数据
    :param seg_data:
    :return:
    """
    sent_list = split_to_sentence(seg_data.text)
    label_list = [[] for _ in range(len(sent_list))]
    for idx, text_sentence in enumerate(sent_list):
        for sent, label in seg_data.label_data.items():
            if sent in text_sentence and label not in label_list[idx]:
                label_list[idx].append(label)
    matrix_data = []
    for label, sent in zip(label_list, sent_list):
        if label:
            matrix_data.append(label + [sent])
        else:
            matrix_data.append(['none', sent])
    return matrix_data


def raw_data_convert_to_step1_data(dataset):
    seg_dataset = []
    for data in dataset:
        ele = dataset_split_to_segment_dataset(data)
        seg_dataset.extend(ele)
    trainable_dataset = []
    for ele in seg_dataset:
        res = segment_data_convert_to_matrix_data(ele)
        trainable_dataset.extend(res)
    return trainable_dataset


def pos_neg_balance(dataset):
    negative_dataset, positive_dataset = [], []

    for data in dataset:
        label = data[0]
        if label == 'none':
            negative_dataset.append(data)
        else:
            positive_dataset.append(data)
    positive_dataset_size = len(positive_dataset)
    negative_dataset_size = len(negative_dataset)
    if positive_dataset_size < negative_dataset_size:
        nega_samples = random.sample(negative_dataset, positive_dataset_size)
        posi_samples = positive_dataset
    else:
        posi_samples = random.sample(positive_dataset, negative_dataset_size)
        nega_samples = negative_dataset

    dataset = posi_samples + nega_samples
    random.shuffle(dataset)
    return dataset


def create_step1_data(train_dataset, test_dataset, step1_train_data_path, step1_test_data_path, label_list_path):
    step1_train_data = raw_data_convert_to_step1_data(train_dataset)
    step1_test_data = raw_data_convert_to_step1_data(test_dataset)

    label_list = []
    for data in step1_train_data + step1_test_data:
        labels = data[:-1]
        for label in labels:
            if label != 'none' and label not in label_list:
                label_list.append(label)

    step1_train_data_after_balance = pos_neg_balance(step1_train_data)

    matrix_to_txt(step1_train_data_path, step1_train_data_after_balance, sep="\t")
    matrix_to_txt(step1_test_data_path, step1_test_data, sep="\t")
    list_to_txt(label_list_path, label_list)


def create_single_file_data(train_dataset, test_dataset, single_file_train_dataset_path, single_file_test_dataset_path):
    # 转为单文书数据集
    single_file_train_dataset = raw_data_convert_to_single_file_data(train_dataset)
    single_file_test_dataset = raw_data_convert_to_single_file_data(test_dataset)

    matrix_to_txt(single_file_train_dataset_path, single_file_train_dataset)
    matrix_to_txt(single_file_test_dataset_path, single_file_test_dataset)


def main(src_path=r"./data/raw_data/SMP-CAIL2021-focus_recognition-train.json",
         single_file_train_dataset_path=r"data/single_file_data/train_data.txt",
         single_file_test_dataset_path=r"data/single_file_data/test_data.txt",
         step1_train_data_path=r"data/step1_data/train_data.txt",
         step1_test_data_path=r"data/step1_data/test_data.txt",
         label_list_path=r"data/step1_data/labels.txt", ):
    json_data = read_json(src_path)
    dataset = json_data['dataset']
    # 原始数据划分训练集和测试集
    train_dataset, test_dataset = raw_data_divide(dataset)
    create_single_file_data(train_dataset, test_dataset, single_file_train_dataset_path, single_file_test_dataset_path)
    create_step1_data(train_dataset, test_dataset, step1_train_data_path, step1_test_data_path, label_list_path)


if __name__ == '__main__':
    fire.Fire(main)
