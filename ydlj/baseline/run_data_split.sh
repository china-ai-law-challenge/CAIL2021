#19年数据下载地址https://github.com/china-ai-law-challenge/CAIL2019
python data_split.py \
          --train_21 data/CAIL2021_train.json \
          --dev_19 data/dev_ground_truth.json \
          --train_19 data/big_train_data.json \
          --train_output data/train.json \
          --dev_output data/dev.json
