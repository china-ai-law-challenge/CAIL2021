"""Evaluate model and calculate results for SMP-CAIL2021-ArgumentationUnderstanding Task1.1.

Author: Yixu GAO yxgao19@fudan.edu.cn
"""
from typing import List
import codecs
import torch

from tqdm import tqdm
from sklearn import metrics
import fire

LABELS = ['1', '2', '3', '4', '5']


def calculate_accuracy_f1(
        golds: List[str], predicts: List[str]) -> tuple:
    """Calculate accuracy and f1 score.

    Args:
        golds: answers
        predicts: predictions given by model

    Returns:
        accuracy, f1 score
    """
    return metrics.accuracy_score(golds, predicts), \
           metrics.f1_score(
               golds, predicts,
               labels=LABELS, average='macro')


def get_labels_from_file(filename):
    """Get labels on the last column from file.

    Args:
        filename: file name

    Returns:
        List[str]: label list
    """
    labels = []
    with codecs.open(filename, 'r', encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            labels.append(line.strip().split(',')[-1])
    return labels


def eval_file(golds_file, predicts_file):
    """Evaluate submission file

    Args:
        golds_file: file path
        predicts_file:  file path

    Returns:
        accuracy, f1 score
    """
    golds = get_labels_from_file(golds_file)
    predicts = get_labels_from_file(predicts_file)
    return calculate_accuracy_f1(golds, predicts)


def evaluate(model, data_loader, device) -> List[str]:
    """Evaluate model on data loader in device.

    Args:
        model: model to be evaluate
        data_loader: torch.utils.data.DataLoader
        device: cuda or cpu

    Returns:
        answer list
    """
    model.eval()
    outputs = torch.tensor([], dtype=torch.float).to(device)
    for batch in tqdm(data_loader, desc='Evaluation', ncols=80):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(*batch)
        outputs = torch.cat([outputs, logits[:, 1]])
    answer_list = []
    for i in range(0, len(outputs), len(LABELS)):
        logits = outputs[i:i + len(LABELS)]
        answer = int(torch.argmax(logits))
        answer_list.append(LABELS[answer])
    return answer_list


def main(golden_file='data/test-1.csv', predict_file='bert-submission-test-1.csv'):
    acc, f1_score = eval_file(golden_file, predict_file)
    print("acc: {}, f1: {}".format(acc, f1_score))

        
if __name__ == '__main__':
    fire.Fire(main)
