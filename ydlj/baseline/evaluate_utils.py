import collections
import json
import logging
import math
import re
import string
from transformers import BasicTokenizer
from dataclasses import dataclass
from typing import List
from data_process_utils import CAILExample


@dataclass
class Prediction:
    """
    用来保存可能的预测结果, feature-result-prediction是一一对应的关系
    """
    feature_index: int
    start_index: int  # 预测的对应Sequence中的开始位置
    end_index: int  # 预测的对应Sequence中的结束位置
    start_logit: float  # 预测的开始得分
    end_logit: float  # 预测的截止得分
    text: str = None  # 对应的在原文中的文本,一开始为None,后面被计算
    orig_start_index: int = None
    orig_end_index: int = None
    final_score: int = None


def compute_predictions(
        all_examples: List[CAILExample],
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        verbose_logging,
        null_score_diff_threshold,
        tokenizer,
        args,
        multi_span_predict: bool = True
):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = []
    all_nbest_predict = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        predictions = []

        min_cls_score = 1000000
        min_cls_score_feature_index = None
        min_cls_start_logits = 0
        min_cls_end_logits = 0

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)

            feature_null_score = result.start_logits[0] + result.end_logits[0]
            if feature_null_score < min_cls_score:
                min_cls_score = feature_null_score
                min_cls_score_feature_index = feature_index
                min_cls_start_logits = result.start_logits[0]
                min_cls_end_logits = result.end_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    predictions.append(
                        Prediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        predictions.append(
            Prediction(
                feature_index=min_cls_score_feature_index,
                start_index=0,
                end_index=0,
                start_logit=min_cls_start_logits,
                end_logit=min_cls_end_logits,
            )
        )

        predictions = sorted(predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
        seen_predictions = {}
        filtered_predictions = []

        for prediction in predictions:
            if len(filtered_predictions) >= n_best_size:
                break

            feature = features[prediction.feature_index]
            if prediction.start_index == 1:
                final_text = "YES"
            elif prediction.start_index == 2:
                final_text = "NO"
            elif prediction.start_index == 0:
                final_text = ""
            else:  # this is a non-null prediction
                tok_tokens = feature.tokens[prediction.start_index: (prediction.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[prediction.start_index]
                orig_doc_end = feature.token_to_orig_map[prediction.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]
                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                prediction.orig_start_index = orig_doc_start
                prediction.orig_end_index = orig_doc_end
            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True
            prediction.text = final_text
            filtered_predictions.append(prediction)
        predictions = filtered_predictions
        if "" not in seen_predictions:
            predictions.append(
                Prediction(
                    feature_index=min_cls_score_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=min_cls_start_logits,
                    end_logit=min_cls_end_logits,
                )
            )
        assert len(predictions) > 0
        if len(predictions) == 1:
            predictions.append(Prediction(feature_index=-1, start_index=-1, end_index=-1,
                                          start_logit=0.0, end_logit=0.0, text="empty"))

        score_normalization(predictions)
        best_non_null_entry = None
        for p in predictions:
            if best_non_null_entry is None and p.text != "":
                best_non_null_entry = p
                break
        score_diff = min_cls_start_logits + min_cls_end_logits - best_non_null_entry.start_logit - best_non_null_entry.end_logit
        predict_answers = []
        if score_diff > null_score_diff_threshold:
            predict_answers = [""]
        else:
            max_score = best_non_null_entry.final_score
            if best_non_null_entry.start_index > 2:
                span_covered = [0 for i in range(len(example.doc_tokens))]
                for p in predictions:
                    if p.start_index > 2 and p.final_score > (max_score * args.multi_span_threshold) \
                            and 1 not in span_covered[p.orig_start_index: (p.orig_end_index + 1)]:
                        predict_answers.append(p.text)
                        span_covered[p.orig_start_index: (p.orig_end_index + 1)] = [1 for i in range(p.orig_start_index, p.orig_end_index + 1)]
            else:
                predict_answers = [best_non_null_entry.text]
        if not multi_span_predict:
            predict_answers = predict_answers[0:1]
        all_predictions.append({"id": example.qas_id, "answer": predict_answers})
        current_nbest = []
        for (i, prediction) in enumerate(predictions):
            output = collections.OrderedDict()
            output["text"] = prediction.text
            output["start_logit"] = prediction.start_logit
            output["end_logit"] = prediction.end_logit
            current_nbest.append(output)
        all_nbest_predict[example.qas_id] = current_nbest

    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False))

    with open(output_nbest_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(all_nbest_predict, indent=4, ensure_ascii=False))

    return all_predictions


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""
    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logging.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logging.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position: (orig_end_position + 1)]
    return output_text


def score_normalization(predictions: List[Prediction]):
    scores = [p.start_logit + p.end_logit for p in predictions]
    max_score = max(scores)
    min_score = min(scores)
    for p in predictions:
        if (max_score - min_score) == 0:
            p.final_score = 0
            continue
        p.final_score = 1.0 * ((p.start_logit + p.end_logit) - min_score) / (max_score - min_score)






