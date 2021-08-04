# -*- encoding: utf-8 -*-

"""
用途      ：
@File    :data_util.py
@Time    :2021/7/16 13:39
@Author  :hyyd
@Software: PyCharm
"""


class DataLoader(object):
    def __init__(self, matrix_data, label_list):
        self.label_list = label_list
        self.data_x = []
        self.data_y = []
        self.init(matrix_data)

    def init(self, matrixs):
        for data_list in matrixs:
            ele = [float(x) for x in data_list[1:]]
            self.data_x.append(ele)

        label_size = len(self.label_list)
        for d in matrixs:
            one_hot_label = [0] * label_size
            label = d[0]
            if label != 'none':
                idx = self.label_list.index(label)
                one_hot_label[idx] = 1
            self.data_y.append(one_hot_label)

    def get_data_x(self):
        return self.data_x

    def get_data_y(self):
        return self.data_y

    def get_label_list(self):
        return self.label_list


class LabelTransform(object):
    def __init__(self, label_list):
        self.labels = label_list

    def multi_hot_to_labels(self, multi_hot_vector):
        """
        multi_hot编码转为标签
        :param multi_hot_vector:一个含有0，1的list
        :return:
        """
        result = []
        for idx, ele in enumerate(multi_hot_vector):
            if ele == 1:
                result.append(self.labels[idx])
        return result

    def label_to_one_hot(self, label):
        one_hot = [0] * len(self.labels)
        if label in self.labels:
            idx = self.labels.index(label)
            one_hot[idx] = 1
        return one_hot



