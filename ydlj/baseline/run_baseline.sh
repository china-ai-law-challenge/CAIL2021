#!bin/bash

date
source ~/.bashrc
module unload gcc
module load gcc/5.5-os7
conda activate rc

python run_cail2021.py \
        --model_path  ./chinese-legal-electra-base-discriminator \  #模型下载地址https://github.com/ymcui/Chinese-ELECTRA
        --train_data_dir ./processed_data/ \
        --eval_data_dir ./processed_data/ \
        --ground_truth_file data/dev.json \
        --output_dir ./output/ \
        --device cuda \
        --batch_size 8 \
        --do_lower_case \
        --logging_steps 200 \
        --save_steps 1000 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 5 \
        --learning_rate 7e-5 \
        --max_answer_length 30 \
        --max_seq_length 512 \
        --max_query_length 64 \
        --doc_stride 128 \
        --multi_span_threshold 0.8 \
        --seed 42
date
