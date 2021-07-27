"""
数据处理相关代码
"""
import argparse
from transformers import ElectraTokenizer
from data_process_utils import *
import gzip
import pickle
import os
from os.path import join


def convert_and_write(args, tokenizer: PreTrainedTokenizer, file, examples_fn, features_fn, is_training):
    logging.info(f"Reading examples from :{file} ...")
    example_list = read_examples(file, is_training=is_training)
    logging.info(f"Total examples:{len(example_list)}")

    logging.info(f"Start converting examples to features.")
    feature_list = convert_examples_to_features(example_list, tokenizer, args, is_training)
    logging.info(f"Total features:{len(feature_list)}")

    logging.info(f"Converting complete, writing examples and features to file.")
    with gzip.open(join(args.output_path, examples_fn), "wb") as file:
        pickle.dump(example_list, file)
    with gzip.open(join(args.output_path, features_fn), "wb") as file:
        pickle.dump(feature_list, file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="The file to be processed."
    )

    parser.add_argument(
        "--for_training",
        action="store_true",
        help="Process for training or not."
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="The prefix of output file's name."
    )

    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to tokenizer which will be used to tokenize text.(ElectraTokenizer)"
    )

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. "
             "Longer will be truncated, and shorter will be padded."
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer will be truncated to the length."
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks."
    )

    parser.add_argument(
        "--output_path",
        default="./processed_data/",
        type=str,
        help="Output path of the constructed examples and features."
    )

    args = parser.parse_args()
    args.max_query_length += 2  # position for token yes and no
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    logging.info("All input parameters:")
    print(json.dumps(vars(args), sort_keys=False, indent=2))

    tokenizer = ElectraTokenizer.from_pretrained(args.tokenizer_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    convert_and_write(args, tokenizer, args.input_file, args.output_prefix + "_examples.pkl.gz",
                      args.output_prefix + "_features.pkl.gz", args.for_training)


if __name__ == "__main__":
    main()


