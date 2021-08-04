"""Data processor for SMP-CAIL2021-ArgumentationUnderstanding Task1.1.

Author: Yixu GAO (yxgao19@fudan.edu.cn)

In data file, each line contains 1 sc sentence and 5 bc sentences.
The data processor convert each line into 5 samples,
each sample with 1 sc sentence and 1 bc sentence.

Usage:
    from data import Data
    # For training, load train and valid set
    # For BERT model
    data = Data('model/bert/vocab.txt', model_type='bert')
    datasets = data.load_train_and_valid_files(
        'SMP-CAIL2021-train.csv', 'SMP-CAIL2021-valid.csv')
    train_set, valid_set_train, valid_set_valid = datasets
    # For testing, load test set
    data = Data('model/bert/vocab.txt', model_type='bert')
    test_set = data.load_file('SMP-CAIL2021-test.csv', train=False)
"""

from typing import List
import jieba
import torch
import pandas as pd
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from tqdm import tqdm


class Data:
    """Data processor for BERT baseline model for SMP-CAIL2021-ArgumentationUnderstanding Task1.1.

    Attributes:
        model_type: 'bert'
        max_seq_len: int, default: 512
        tokenizer:  BertTokenizer for bert
    """
    def __init__(self,
                 vocab_file='',
                 max_seq_len: int = 512,
                 model_type: str = 'bert'):
        """Initialize data processor for SMP-CAIL2021-Argmine.
        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
            model_type: 'bert'
        """
        self.model_type = model_type
        self.tokenizer = BertTokenizer(vocab_file)
        self.max_seq_len = max_seq_len

    def load_file(self,
                  file_path='SMP-CAIL2021-train.csv',
                  train=True) -> TensorDataset:
        """Load train file and construct TensorDataset.

        Args:
            file_path: train file with last column as label
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label

        Returns:
            BERT model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
        """
        sc_list, bc_list, label_list = self._load_file(file_path, train)
        dataset = self._convert_sentence_pair_to_bert_dataset(
                sc_list, bc_list, label_list)
        return dataset

    def load_train_and_valid_files(self, train_file, valid_file):
        """Load all files for SMP-CAIL2021-ArgumentationUnderstanding Task1.1.

        Args:
            train_file, valid_file: files for SMP-CAIL2021-ArgumentationUnderstanding Task1.1.

        Returns:
            train_set, valid_set_train, valid_set_valid
            all are torch.utils.data.TensorDataset
        """
        print('Loading train records for train...')
        train_set = self.load_file(train_file, True)
        print(len(train_set), 'training records loaded.')
        print('Loading train records for valid...')
        valid_set_train = self.load_file(train_file, False)
        print(len(valid_set_train), 'train records loaded.')
        print('Loading valid records...')
        valid_set_valid = self.load_file(valid_file, False)
        print(len(valid_set_valid), 'valid records loaded.')
        return train_set, valid_set_train, valid_set_valid

    def _load_file(self, filename, train: bool = True):
        """Load train/test file.

        For train file,
        The ratio between positive samples and negative samples is 1:4
        Copy positive 3 times so that positive:negative = 1:1

        Args:
            filename: SMP-CAIL2021-ArgumentationUnderstanding file
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label

        Returns:
            sc_list, bc_list, label_list with the same length
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: List[int], list of labels
        """
        data_frame = pd.read_csv(filename)
        sc_list, bc_list, label_list = [], [], []
        for row in data_frame.itertuples(index=False):
            candidates = row[3:8]
            answer = int(row[-1]) if train else None
            sc_tokens = self.tokenizer.tokenize(row[2])
            for i, _ in enumerate(candidates):
                bc_tokens = self.tokenizer.tokenize(candidates[i])
                if train:
                    if i + 1 == answer:
                        # Copy positive sample 4 times
                        for _ in range(len(candidates) - 1):
                            sc_list.append(sc_tokens)
                            bc_list.append(bc_tokens)
                            label_list.append(1)
                    else:
                        sc_list.append(sc_tokens)
                        bc_list.append(bc_tokens)
                        label_list.append(0)
                else:  # test
                    sc_list.append(sc_tokens)
                    bc_list.append(bc_tokens)
        return sc_list, bc_list, label_list

    def _convert_sentence_pair_to_bert_dataset(
            self, s1_list, s2_list, label_list=None):
        """Convert sentence pairs to dataset for BERT model.

        Args:
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
        """
        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for i, _ in tqdm(enumerate(s1_list), ncols=80):
            tokens = ['[CLS]'] + s1_list[i] + ['[SEP]']
            segment_ids = [0] * len(tokens)
            tokens += s2_list[i] + ['[SEP]']
            segment_ids += [1] * (len(s2_list[i]) + 1)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
                segment_ids = segment_ids[:self.max_seq_len]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            tokens_len = len(input_ids)
            input_ids += [0] * (self.max_seq_len - tokens_len)
            segment_ids += [0] * (self.max_seq_len - tokens_len)
            input_mask += [0] * (self.max_seq_len - tokens_len)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # test
        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids)


def test_data():
    """Test for data module."""
    # For BERT model
    data = Data('model/bert/vocab.txt', model_type='bert')
    _, _, _ = data.load_train_and_valid_files(
        'SMP-CAIL2021-train.csv',
        'SMP-CAIL2021-test1.csv')


if __name__ == '__main__':
    test_data()
