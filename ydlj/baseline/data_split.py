"""
用于对数据划分，使用19年和21年训练集（8/9）得到新的训练集，使用19年开发集和21年训练集（1/9）得到新的开发集
"""
from data_process_utils import write_example_orig_file
from evaluate_2021 import *
import random
import argparse
import logging
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_21",
        type=str,
        required=True,
        help="The train set of CAIL 2021."
    )
    parser.add_argument(
        "--dev_19",
        type=str,
        required=True,
        help="The dev set of CAIL 2019."
    )
    parser.add_argument(
        "--train_19",
        type=str,
        required=True,
        help="The train set of CAIL 2019."
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="data/train.json",
        help="Directory to write train set after split."
    )
    parser.add_argument(
        "--dev_output",
        type=str,
        default="data/dev.json",
        help="Directory to write dev set after split."
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logging.info("All input parameters:")
    print(json.dumps(vars(args), sort_keys=False, indent=2))

    # if not os.path.exists(args.output_path):
    #     os.makedirs(args.output_path)

    examples_train_21 = read_examples(args.train_21, is_training=True)
    examples_dev_19 = read_examples(args.dev_19, is_training=True)
    examples_train_19 = read_examples(args.train_19, is_training=True)

    random.shuffle(examples_train_21)
    random.shuffle(examples_dev_19)

    # split CAIL 2021 train-set to 8 : 1
    num_per_qa_dev = int(1.0 * len(examples_train_21) * (1 / 9))
    logging.info(f"num per qa type in dev: {num_per_qa_dev}")

    train_set = []
    dev_set = []

    dev_set.extend(examples_train_21[:num_per_qa_dev])
    train_set.extend(examples_train_21[num_per_qa_dev:])
    train_set.extend(examples_train_19)

    yes_examples_19 = [e for e in examples_dev_19 if get_example_qa_type(e) == TYPE_YES]
    no_examples_19 = [e for e in examples_dev_19 if get_example_qa_type(e) == TYPE_NO]
    single_span_examples_19 = [e for e in examples_dev_19 if get_example_qa_type(e) == TYPE_SINGLE_SPAN]
    no_answer_examples_19 = [e for e in examples_dev_19 if get_example_qa_type(e) == TYPE_NO_ANSWER]

    dev_set.extend(yes_examples_19[:num_per_qa_dev])
    dev_set.extend(no_examples_19[:num_per_qa_dev])
    dev_set.extend(single_span_examples_19[:num_per_qa_dev])
    dev_set.extend(no_answer_examples_19[:num_per_qa_dev])

    logging.info(f"Train set example num: {len(train_set)}")
    logging.info(f"Test set example num: {len(dev_set)}")

    random.shuffle(train_set)
    random.shuffle(dev_set)

    # re-generate id for example to prevent duplicate id between 19 and 21 data
    for i, example in enumerate(train_set):
        example.qas_id = i
    for i, example in enumerate(dev_set):
        example.qas_id = i

    # write split data back to file
    write_example_orig_file(train_set, args.train_output)
    write_example_orig_file(dev_set, args.dev_output)

    logging.info("Complete!")


if __name__ == "__main__":
    main()




