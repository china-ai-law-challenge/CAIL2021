# -*- encoding: utf-8 -*-

"""
用途      ：工具类
@File    :utils.py
@Time    :2021/7/16 15:41
@Author  :Chenhao Zhai
@Software: PyCharm
"""
import json
import os
import pickle
import re


def mkdir(root):
    """
    给定一个路径，递归创建路径中的目录。
    如果给到的root不是一个dir_path而是一个file_path则提取相应的路径名进行文件夹的创建
    :param root:路径名
    :return:
    """
    if not os.path.isdir(root):
        # 不是目录的路径，则提取路径中的目录
        root = os.path.split(root)[0]
    if not os.path.exists(root):
        # 不存在则创建路径中的目录
        os.makedirs(root)


def read_pickle(path):
    """
    读pickle文件
    :param path:pickle文件路径
    :return:
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(path, data):
    """
    写pickle文件
    :param path:文件路径
    :param data:数据
    :return:
    """
    mkdir(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def txt_to_list(file_path):
    """
    按行读取txt中的内容
    :param file_path: 路径地址
    :return:
    """
    with open(file_path, "r", encoding="utf-8") as fin:
        return [row_data.strip() for row_data in fin.readlines()]


def list_to_txt(file_path, data):
    """
    list数据，按行存放到txt中
    :param file_path:
    :param data:
    :return:
    """
    mkdir(file_path)
    with open(file_path, "w", encoding="utf-8") as fout:
        for row_data in data:
            fout.write(str(row_data) + "\n")


def matrix_to_txt(file_path, matrix, sep=" "):
    """
    二维数组按照矩阵的方式存入txt中
    :param file_path:
    :param matrix:
    :return:
    """
    """二维列表存储到txt中"""
    mkdir(file_path)
    with open(file_path, "w", encoding='utf-8') as fout:
        for row_data in matrix:
            fout.write(sep.join([str(ele) for ele in row_data]) + "\n")


def txt_to_matrix(file_path, sep=" "):
    """
    矩阵的方式存储的内容读取成为二维列表
    :param file_path:
    :return:
    """
    matrix = []
    with open(file_path, "r", encoding='utf-8') as fin:
        for row_data in fin.readlines():
            matrix.append(row_data.strip().split())
    return matrix


def read_json(path):
    """
    读取json文件
    :param path:
    :return:
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    """
    读取json文件
    :param path:
    :return:
    """
    mkdir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=False, indent=4)


def split_to_sentence(paragraph, half=False):
    """
    将段落切分为句子
    :param paragraph: 代表输入的段落
    :param half: 如果为True则切分时以半句为单位
                 如果为False，切分时以整句为单位
    :return:
    """
    pattern = r"。|？|！|……|,|;|，|；" if half else r"。|？|！|……"
    return [ele for ele in re.split(pattern, paragraph) if ele]


def remove_blank(text):
    """移除空白，不仅仅是空格，还包括\n\t等"""
    return "".join(str(text).split())
