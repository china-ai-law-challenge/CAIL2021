import json
import torch
from torch.utils.data import Dataset
from utils import get_fact, label2idx


class CaseData(Dataset):
    def __init__(self, file_name, class_num):
        self.data = json.load(open(file_name, encoding='utf-8'))
        self.class_num = class_num

    def __getitem__(self, idx):
        fact = get_fact(self.data[idx]['content'])

        label = torch.zeros(self.class_num)
        for i in label2idx(self.data[idx]['result']):
            label[i] = 1

        return fact, label

    def __len__(self):
        return len(self.data)
