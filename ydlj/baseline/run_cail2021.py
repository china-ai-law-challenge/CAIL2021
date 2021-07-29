import argparse
from transformers import AutoModel, AutoConfig,  AutoTokenizer
from model import MultiSpanQA
from data_process_utils import *
import torch
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pickle
import gzip
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import torch.nn as nn
from tqdm import trange, tqdm
import timeit
import os
from evaluate_utils import compute_predictions
from evaluate_2021 import CJRCEvaluator
from os.path import join


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def load_data(file_path):
    """
    Used to load CAILExample and CAILFeature objects.
    """
    return pickle.load(gzip.open(file_path, "rb"))


def train(args, train_dataset: TensorDataset, model: nn.Module,
          tokenizer):
    args.train_batch_size = args.batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logging.info("***** Running training *****")
    logging.info("  Total training samples number = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logging.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
        )
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch",
    )
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_labels": batch[3],
                "end_labels": batch[4],
            }
            outputs = model(**inputs)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            logging_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging.info(f"Current global step: {global_step}, start evaluating!")
                    logging.info(f"average loss of batch: {logging_loss / args.logging_steps}")
                    logging_loss = 0
                    results = evaluate(args, model, tokenizer, prefix=f"{global_step}-dev",
                                       eval_data_dir=args.eval_data_dir, ground_truth_file=args.ground_truth_file,
                                       multi_span_predict=True)
                    logging.info(f"Evaluation result in dev-set at global step {global_step}: {results}")

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model
                    torch.save(model_to_save, os.path.join(output_dir, "checkpoint.bin"))
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logging.info("Saving model checkpoint to %s", output_dir)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix, eval_data_dir: str,
             ground_truth_file: str = None, multi_span_predict=True):
    logging.info(f"Loading data for evaluation from {eval_data_dir}!")
    examples = load_data(join(args.eval_data_dir, "dev_examples.pkl.gz"))
    features = load_data(join(args.eval_data_dir, "dev_features.pkl.gz"))
    dataset = convert_features_to_dataset(features, is_training=False)
    logging.info("Complete Loading!")

    args.eval_batch_size = args.batch_size
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logging.info("***** Running evaluation {} *****".format(prefix))
    logging.info("  Num examples = %d", len(dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            features_indexes = batch[6]
            outputs = model(**inputs)

        for i, feature_index in enumerate(features_indexes):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [to_list(o[i]) for o in outputs]

            start_logits, end_logits = output

            result = CAILResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits,
            )
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logging.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    compute_predictions(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        False,
        args.null_score_diff_threshold,
        tokenizer,
        args,
        multi_span_predict=multi_span_predict
    )
    if ground_truth_file is not None:
        evaluator = CJRCEvaluator(ground_truth_file)
        pred_data = CJRCEvaluator.preds_to_dict(output_prediction_file)
        res = evaluator.model_performance(pred_data)
        return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path to Electra model."
    )

    parser.add_argument(
        "--train_data_dir",
        default=None,
        required=True,
        type=str,
        help="The directory which contain the generated train-set examples and features file."
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory of checkpoints and predictions."
    )

    parser.add_argument(
        "--eval_data_dir",
        default=None,
        type=str,
        required=True,
        help="The directory which contain the generated dev-set examples and features file."
    )

    parser.add_argument(
        "--ground_truth_file",
        default=None,
        type=str,
        help="The ground truth file of dev-set."
    )

    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )

    parser.add_argument("--device", default="cuda", type=str, help="Whether not to use CUDA when available")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training and evaluating.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization.")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=600, help="Save checkpoint every X updates steps.")

    parser.add_argument(
        "--max_answer_length",
        default=32,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
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
        "--multi_span_threshold",
        type=float,
        default=0.8,
        help="Span which score is bigger than (max_span_score * multi_span_threshold) will also be output!"
    )

    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logging.info("All input parameters:")
    print(json.dumps(vars(args), sort_keys=False, indent=2))

    set_seed(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pretrain_model = AutoModel.from_pretrained(args.model_path, config=config)
    model = MultiSpanQA(pretrain_model)

    model.to(device=args.device)

    logging.info("Loading pre-processed examples and features!")
    train_examples = load_data(args.train_data_dir + "/train_examples.pkl.gz")
    train_features = load_data(args.train_data_dir + "/train_features.pkl.gz")
    logging.info("Complete loading!")

    logging.info("Converting features to pytorch Dataset!")
    train_dataset = convert_features_to_dataset(train_features, is_training=True)
    logging.info("Complete converting!")

    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logging.info("global_step = %s, average loss = %s", global_step, tr_loss)
    logging.info("Training Complete!")

    logging.info("Saving model checkpoint to %s", args.output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save, os.path.join(args.output_dir, "checkpoint.bin"))
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    tokenizer.save_pretrained(args.output_dir)
    results = evaluate(args, model, tokenizer, prefix=f"final-eval",
                       eval_data_dir=args.eval_data_dir, ground_truth_file=args.ground_truth_file,
                       multi_span_predict=True)
    logging.info(f"Final evaluate results on dev-set: {results}")


if __name__ == "__main__":
    main()



