"""Separate train and valid set for train file.

Author: Yixu GAO yxgao19@fudan.edu.cn

Usage:
    python prepare.py --train_in 'data/SMP-CAIL2021-train.csv' \
                      --train_out 'data/train.csv' \
                      --valid_out 'data/valid.csv'
"""
import fire
import pandas as pd


def main(train_in='data/SMP-CAIL2021-train.csv',
         train_out='data/train.csv',
         valid_out='data/valid.csv'):
    """Main method to divide dataset.

    Args:
        train_in: origin train file
        train_out: train file
        valid_out: valid file
    """
    data = pd.read_csv(train_in, encoding='utf-8')
    total_num = data.shape[0]
    train_num = int(0.8 * total_num)
    data[:train_num].to_csv(train_out, encoding='utf-8', index=False)
    data[train_num:].to_csv(valid_out, encoding='utf-8', index=False)


if __name__ == '__main__':
    fire.Fire(main)
