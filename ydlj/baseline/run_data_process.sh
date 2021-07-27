source ~/.bashrc
conda activate rc
electra_path="" #模型下载地址https://github.com/ymcui/Chinese-ELECTRA
python data_process.py \
          --input_file data/train.json \
          --for_training \
          --output_prefix train \
          --do_lower_case \
          --tokenizer_path $electra_path \
          --max_seq_length 512 \
          --max_query_length 64 \
          --doc_stride 128 \
          --output_path ./processed_data/

python data_process.py \
          --input_file data/dev.json \
          --output_prefix dev \
          --do_lower_case \
          --tokenizer_path $electra_path \
          --max_seq_length 512 \
          --max_query_length 64 \
          --doc_stride 128 \
          --output_path ./processed_data/