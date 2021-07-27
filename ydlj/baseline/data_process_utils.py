import logging
from dataclasses import dataclass
from typing import List, Dict
import json
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BasicTokenizer
from transformers.tokenization_utils import _is_whitespace, _is_punctuation, _is_control
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

YES_TOKEN = "[unused1]"
NO_TOKEN = "[unused2]"


class CAILExample:
    def __init__(self,
                 qas_id: str,
                 question_text: str,
                 context_text: str,
                 answer_texts: List[str],
                 answer_start_indexes: List[int],
                 is_impossible: bool,
                 is_yes_no: bool,
                 is_multi_span: bool,
                 answers: List,
                 case_id: str,
                 case_name: str):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_texts = answer_texts
        self.answer_start_indexes = answer_start_indexes
        self.is_impossible = is_impossible
        self.is_yes_no = is_yes_no
        self.is_multi_span = is_multi_span
        self.answers = answers
        self.case_id = case_id
        self.case_name = case_name

        self.doc_tokens = []
        self.char_to_word_offset = []

        raw_doc_tokens = customize_tokenizer(context_text, True)
        k = 0
        temp_word = ""
        for char in self.context_text:
            if _is_whitespace(char):
                self.char_to_word_offset.append(k - 1)
                continue
            else:
                temp_word += char
                self.char_to_word_offset.append(k)
            if temp_word.lower() == raw_doc_tokens[k]:
                self.doc_tokens.append(temp_word)
                temp_word = ""
                k += 1
        assert k == len(raw_doc_tokens)

        if answer_texts is not None:  # if for training
            start_positions = []
            end_positions = []

            if not is_impossible and not is_yes_no:
                for i in range(len(answer_texts)):
                    answer_offset = context_text.index(answer_texts[i])
                    answer_length = len(answer_texts[i])
                    start_position = self.char_to_word_offset[answer_offset]
                    end_position = self.char_to_word_offset[answer_offset + answer_length - 1]
                    start_positions.append(start_position)
                    end_positions.append(end_position)
            else:
                start_positions.append(-1)
                end_positions.append(-1)
            self.start_positions = start_positions
            self.end_positions = end_positions

    def __repr__(self):
        string = ""
        for key, value in self.__dict__.items():
            string += f"{key}: {value}\n"
        return f"<{self.__class__}>"


@dataclass
class CAILFeature:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    cls_index: int
    p_mask: List
    example_index: int
    unique_id: int
    paragraph_len: int
    token_is_max_context: object
    tokens: List
    token_to_orig_map: Dict
    start_positions: List[int]
    end_positions: List[int]
    is_impossible: bool


@dataclass
class CAILResult:
    unique_id: int
    start_logits: torch.Tensor
    end_logits: torch.Tensor


def read_examples(file: str, is_training: bool) -> List[CAILExample]:
    example_list = []
    with open(file, "r", encoding="utf-8") as file:
        original_data = json.load(file)["data"]

    for entry in tqdm(original_data, disable=True):
        case_id = entry["caseid"]
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            case_name = paragraph["casename"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                qas_id = qa["id"]
                answer_texts = None
                answer_starts = None
                is_impossible = None
                is_yes_no = None
                is_multi_span = None
                all_answers = None
                if is_training:
                    all_answers = qa["answers"]
                    if len(all_answers) == 0:
                        answer = []
                    else:
                        answer = all_answers[0]
                    # a little difference between 19 and 21 data.
                    if type(answer) == dict:
                        answer = [answer]

                    if len(answer) == 0:  # NO Answer
                        answer_texts = [""]
                        answer_starts = [-1]
                    else:
                        answer_texts = []
                        answer_starts = []
                        for a in answer:
                            answer_texts.append(a["text"])
                            answer_starts.append(a["answer_start"])
                    # Judge YES or NO
                    if len(answer_texts) == 1 and answer_starts[0] == -1 and (answer_texts[0] == "YES" or answer_texts[0] == "NO"):
                        is_yes_no = True
                    else:
                        is_yes_no = False
                    # Judge Multi Span
                    if len(answer_texts) > 1:
                        is_multi_span = True
                    else:
                        is_multi_span = False
                    # Judge No Answer
                    if len(answer_texts) == 1 and answer_texts[0] == "":
                        is_impossible = True
                    else:
                        is_impossible = False

                example = CAILExample(
                    qas_id=qas_id,
                    question_text=question,
                    context_text=context,
                    answer_texts=answer_texts,
                    answer_start_indexes=answer_starts,
                    is_impossible=is_impossible,
                    is_yes_no=is_yes_no,
                    is_multi_span=is_multi_span,
                    answers=all_answers,
                    case_id=case_id,
                    case_name=case_name
                )
                # Discard possible bad example
                if is_training and example.answer_start_indexes[0] >= 0:
                    for i in range(len(example.answer_texts)):
                        actual_text = "".join(example.doc_tokens[example.start_positions[i]: (example.end_positions[i] + 1)])
                        cleaned_answer_text = "".join(whitespace_tokenize(example.answer_texts[i]))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logging.info(f"Could not find answer: {actual_text} vs. {cleaned_answer_text}")
                            continue
                example_list.append(example)
    return example_list


def convert_examples_to_features(example_list: List[CAILExample], tokenizer: PreTrainedTokenizer, args,
                                 is_training: bool) -> List[CAILFeature]:
    # Validate there are no duplicate ids in example_list
    qas_id_set = set()
    for example in example_list:
        if example.qas_id in qas_id_set:
            raise Exception("Duplicate qas_id!")
        else:
            qas_id_set.add(example.qas_id)

    feature_list = []
    unique_id = 0
    example_index = 0
    i = 0
    for example in tqdm(example_list, disable=True):
        i += 1
        if i % 100 == 0:
            print(i)
        current_example_features = convert_single_example_to_features(example, tokenizer, args.max_seq_length,
                                                                      args.max_query_length, args.doc_stride, is_training)
        for feature in current_example_features:
            feature.example_index = example_index
            feature.unique_id = unique_id
            unique_id += 1
        example_index += 1
        feature_list.extend(current_example_features)

    return feature_list


def convert_single_example_to_features(example: CAILExample, tokenizer: PreTrainedTokenizer,
                                       max_seq_length, max_query_length, doc_stride, is_training) -> List[CAILFeature]:
    """
    Transfer original text to sequence which can be accepted by ELECTRA
    Format: [CLS] YES_TOKEN NO_TOKEN question [SEP] context [SEP]
    """
    features = []
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training:
        if example.is_impossible or example.answer_start_indexes[0] == -1:
            start_positions = [-1]
            end_positions = [-1]
        else:
            start_positions = []
            end_positions = []
            for i in range(len(example.start_positions)):
                start_position = orig_to_tok_index[example.start_positions[i]]
                if example.end_positions[i] < len(example.doc_tokens) - 1:
                    end_position = orig_to_tok_index[example.end_positions[i] + 1] - 1
                else:
                    end_position = len(all_doc_tokens) - 1
                (start_position, end_position) = _improve_answer_span(
                    all_doc_tokens, start_position, end_position, tokenizer, example.answer_texts[i]
                )
                start_positions.append(start_position)
                end_positions.append(end_position)
    else:
        start_positions = None
        end_positions = None

    query_tokens = tokenizer.tokenize(example.question_text)
    query_tokens = [YES_TOKEN, NO_TOKEN] + query_tokens
    truncated_query = tokenizer.encode(query_tokens, add_special_tokens=False, max_length=max_query_length, truncation=True)

    sequence_pair_added_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    assert sequence_pair_added_tokens == 3

    added_tokens_num_before_second_sequence = tokenizer.num_special_tokens_to_add(pair=False)
    assert added_tokens_num_before_second_sequence == 2
    span_doc_tokens = all_doc_tokens
    spans = []
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(
            truncated_query,
            span_doc_tokens,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            padding="max_length",
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            truncation="only_second",
            return_token_type_ids=True
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids = encoded_dict["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        token_to_orig_map[0] = -1
        token_to_orig_map[1] = -1
        token_to_orig_map[2] = -1

        token_is_max_context = {0: True, 1: True, 2: True}
        for i in range(paragraph_len):
            index = len(truncated_query) + added_tokens_num_before_second_sequence + i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + added_tokens_num_before_second_sequence
        encoded_dict["token_is_max_context"] = token_is_max_context
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        encoded_dict["token_type_ids"][1] = 1
        encoded_dict["token_type_ids"][2] = 1

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or len(encoded_dict["overflowing_tokens"]) == 0:
            break
        else:
            span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        p_mask = np.array(span["token_type_ids"])
        p_mask = np.minimum(p_mask, 1)
        p_mask = 1 - p_mask
        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1
        p_mask[cls_index] = 0
        p_mask[1] = 0
        p_mask[2] = 0

        current_start_positions = None
        current_end_positions = None
        span_is_impossible = None
        if is_training:
            current_start_positions = [0 for i in range(max_seq_length)]
            current_end_positions = [0 for i in range(max_seq_length)]
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            doc_offset = len(truncated_query) + added_tokens_num_before_second_sequence
            for i in range(len(start_positions)):
                start_position = start_positions[i]
                end_position = end_positions[i]
                if start_position >= doc_start and end_position <= doc_end:
                    span_is_impossible = False
                    current_start_positions[start_position - doc_start + doc_offset] = 1
                    current_end_positions[end_position - doc_start + doc_offset] = 1

            if example.is_yes_no:
                assert len(example.answer_start_indexes) == 1
                assert 1 not in current_start_positions and 1 not in current_end_positions
                if example.answer_texts[0] == "YES" and example.answer_start_indexes[0] == -1:
                    current_start_positions[1] = 1
                    current_end_positions[1] = 1
                elif example.answer_texts[0] == "NO" and example.answer_start_indexes[0] == -1:
                    current_start_positions[2] = 1
                    current_end_positions[2] = 1
                else:
                    raise Exception("example构造出错,请检查")
                span_is_impossible = False

            if 1 not in current_start_positions:  # Current Feature does not contain answer span
                span_is_impossible = True
                current_start_positions[cls_index] = 1
                current_end_positions[cls_index] = 1
            assert span_is_impossible is not None
        features.append(
            CAILFeature(
                input_ids=span["input_ids"],
                attention_mask=span["attention_mask"],
                token_type_ids=span["token_type_ids"],
                cls_index=cls_index,
                p_mask=p_mask.tolist(),
                example_index=0,
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_positions=current_start_positions,
                end_positions=current_end_positions,
                is_impossible=span_is_impossible
            )
        )
    return features


def convert_features_to_dataset(features: List[CAILFeature], is_training: bool) -> Dataset:
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_example_indexes = torch.tensor([f.example_index for f in features], dtype=torch.long)
    all_feature_indexes = torch.arange(all_input_ids.size(0), dtype=torch.long)
    if is_training:
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        all_start_labels = torch.tensor([f.start_positions for f in features], dtype=torch.float)
        all_end_labels = torch.tensor([f.end_positions for f in features], dtype=torch.float)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_start_labels,
            all_end_labels,
            all_cls_index,
            all_p_mask,
            all_is_impossible,
            all_example_indexes,
            all_feature_indexes
        )
    else:
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_cls_index,
            all_p_mask,
            all_example_indexes,
            all_feature_indexes
        )
    return dataset


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """
    Check if this is the 'max context' doc span for the token.
    """
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """
    Returns tokenized answer spans that better match the annotated answer.
    没太明白,原本答案文本和input_start到input_end之间对应的原始文本不应该本就对应吗,为什么还可能找出一个对应的子区间?...
    """
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def customize_tokenizer(text: str, do_lower_case=True) -> List[str]:
    temp_x = ""
    for char in text:
        if _is_chinese_char(ord(char)) or _is_punctuation(char) or _is_whitespace(char) or _is_control(char):
            temp_x += " " + char + " "
        else:
            temp_x += char
    if do_lower_case:
        temp_x = temp_x.lower()
    return temp_x.split()


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def whitespace_tokenize(text: str):
    if text is None:
        return []
    text = text.strip()
    tokens = text.split()
    return tokens


def write_example_orig_file(examples: List[CAILExample], file: str):
    """
    convert examples to original json file
    """
    data_list = []
    for example in examples:
        data = {
            "paragraphs": [
                {
                    "context": example.context_text,
                    "casename": example.case_name,
                    "qas": [
                        {
                            "question": example.question_text,
                            "answers": example.answers,
                            "id": example.qas_id,
                            "is_impossible": "true" if example.is_impossible else "false",
                        }
                    ]
                }
            ],
            "caseid": example.case_id
        }
        data_list.append(data)
    final_data = {
        "data": data_list,
        "version": "1.0"
    }
    with open(file, mode="w", encoding="utf-8") as file:
        file.write(json.dumps(final_data, ensure_ascii=False))

