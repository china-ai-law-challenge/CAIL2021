# -*- encoding: utf-8 -*-

"""
用途      ：整体流程输出的测试
@File    :piplein_evaluate.py
@Time    :2021/7/16 16:27
@Author  :hyyd
@Software: PyCharm
"""
from utils import *
import fire


def cal_score(predictions, label_answer):
    """
    计算最终得分
    :param predictions: 预测结果二维列表
    :param label_answer: 标签一维列表
    :return:
    """
    top1_acc, top3_acc = [], []
    for pred_list, label_list in zip(predictions, label_answer):
        if label_list == 'none':
            if not pred_list:
                top1_acc.append(1)
                top3_acc.append(1)
            else:
                top1_acc.append(0)
                top3_acc.append(0)
            continue

        if label_list in pred_list[:1]:
            top1_acc.append(1)
        else:
            top1_acc.append(0)

        if label_list in pred_list[:3]:
            top3_acc.append(1)
        else:
            top3_acc.append(0)
    top1_acc_val = sum(top1_acc) / len(top1_acc)
    top3_acc_val = sum(top3_acc) / len(top3_acc)
    final_score = 0.7 * top1_acc_val + 0.3 * top3_acc_val
    print("top1_acc:", top1_acc_val)
    print("top3_acc:", top3_acc_val)
    print("final_score:", final_score)
    return final_score


def main(prediction_path=r"./data/output/prediction.txt",
         label_path=r"./data/output/criterion.txt"):
    prediction_matrix = txt_to_matrix(prediction_path)
    label_list = txt_to_list(label_path)
    score = cal_score(prediction_matrix, label_list)
    return score


if __name__ == '__main__':
    fire.Fire(main)
