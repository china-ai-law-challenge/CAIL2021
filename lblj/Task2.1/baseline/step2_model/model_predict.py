# -*- encoding: utf-8 -*-

"""
用途      ：模型预测输出
@File    :model_predict.py
@Time    :2021/7/16 14:01
@Author  :hyyd
@Software: PyCharm
"""
import sys

sys.path.append("..")
from utils import *


class MultiLabelPredictModel(object):

    def __init__(self, model_save_path):
        self.model = read_pickle(model_save_path)

    def predict(self, np_float_list):
        return self.model.predict(np_float_list)
