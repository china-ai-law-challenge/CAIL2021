
import json
import sys

from collections import Counter, OrderedDict
from data_process_utils import CAILExample, read_examples


OPTS = None

TYPE_YES = "Yes"
TYPE_NO = "No"
TYPE_NO_ANSWER = "No Answer"
TYPE_SINGLE_SPAN = "Single Span"
TYPE_MULTI_SPAN = "Multi Span"


class CJRCEvaluator:
    def __init__(self, gold_file):
        examples = read_examples(gold_file, True)
        id_to_example = {}
        for example in examples:
            id_to_example[example.qas_id] = example
        self.id_to_example = id_to_example
        self.gold_data = CJRCEvaluator.gold_answers_to_dict(gold_file)

    @staticmethod
    def gold_answers_to_dict(gold_file):
        dataset = json.load(open(gold_file, mode="r", encoding="utf-8"))
        gold_dict = {}
        # id_to_domain = {}
        for story in dataset['data']:
            qas = story["paragraphs"][0]["qas"]
            for qa in qas:
                qid = qa['id']
                gold_answers = []
                answers = qa["answers"]
                if len(answers) == 0:
                    gold_answers = ['']
                else:
                    for answer in qa["answers"]:
                        if type(answer) == dict:
                            gold_answers.append(answer["text"])
                        elif type(answer) == list:
                            gold_answers.append("".join([a["text"] for a in answer]))
                if qid in gold_dict:
                    sys.stderr.write("Gold file has duplicate stories: {}".format(qid))
                gold_dict[qid] = gold_answers
        return gold_dict

    @staticmethod
    def preds_to_dict(pred_file):
        preds = json.load(open(pred_file, mode="r", encoding="utf-8"))
        pred_dict = {}
        for pred in preds:
            pred_dict[pred['id']] = "".join(pred['answer'])
        return pred_dict

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, storys and extra whitespace."""

        def remove_punc(text):
            return "".join(ch for ch in text if ch.isdigit() or ch.isalpha())

        def lower(text):
            return text.lower()
    
        return remove_punc(lower(s))

    @staticmethod
    def get_tokens(s):
        if not s: return []
        return list(CJRCEvaluator.normalize_answer(s))

    @staticmethod
    def compute_exact(a_gold, a_pred):
        return int(CJRCEvaluator.normalize_answer(a_gold) == CJRCEvaluator.normalize_answer(a_pred))

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = CJRCEvaluator.get_tokens(a_gold)
        pred_toks = CJRCEvaluator.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def _compute_turn_score(a_gold_list, a_pred):
        f1_sum = 0.0
        em_sum = 0.0
        if len(a_gold_list) > 1:
            for i in range(len(a_gold_list)):
                # exclude the current answer
                gold_answers = a_gold_list[0:i] + a_gold_list[i + 1:]
                em_sum += max(CJRCEvaluator.compute_exact(a, a_pred) for a in gold_answers)
                f1_sum += max(CJRCEvaluator.compute_f1(a, a_pred) for a in gold_answers)
        else:
            em_sum += max(CJRCEvaluator.compute_exact(a, a_pred) for a in a_gold_list)
            f1_sum += max(CJRCEvaluator.compute_f1(a, a_pred) for a in a_gold_list)
        if f1_sum != 1:
            a = 1 + 1
        return {'em': em_sum / max(1, len(a_gold_list)), 'f1': f1_sum / max(1, len(a_gold_list))}

    @staticmethod
    def _compute_turn_score1(a_gold_list, a_pred):
        f1 = 0.0
        em = 0.0
        if len(a_gold_list) > 1:
            for i in range(len(a_gold_list)):
                # exclude the current answer
                em = max(em, CJRCEvaluator.compute_exact(a_gold_list[i], a_pred))
                f1 = max(f1, CJRCEvaluator.compute_f1(a_gold_list[i], a_pred))
        else:
            em = max(CJRCEvaluator.compute_exact(a, a_pred) for a in a_gold_list)
            f1 = max(CJRCEvaluator.compute_f1(a, a_pred) for a in a_gold_list)
        if em != 1 or f1 != 1:
            a = 1 + 1
        return {'em': em, 'f1': f1}

    def compute_turn_score(self, qid, a_pred):
        ''' This is the function what you are probably looking for. a_pred is the answer string your model predicted. '''
        a_gold_list = self.gold_data[qid]
        return CJRCEvaluator._compute_turn_score(a_gold_list, a_pred)

    def get_raw_scores(self, pred_data):
        ''''Returns a dict with score'''
        exact_scores = {}
        f1_scores = {}
        for qid in self.gold_data:
            if qid not in pred_data:
                sys.stderr.write('Missing prediction for {}\n'.format(qid))
                continue
            a_pred = pred_data[qid]
            scores = self.compute_turn_score(qid, a_pred)
            # Take max over all gold answers
            exact_scores[qid] = scores['em']
            f1_scores[qid] = scores['f1']
        return exact_scores, f1_scores

    def get_raw_scores_human(self):
        '''
        Returns a dict with score
        '''
        exact_scores = {}
        f1_scores = {}
        for qid in self.gold_data:
            f1_sum = 0.0
            em_sum = 0.0
            if len(self.gold_data[qid]) > 1:
                for i in range(len(self.gold_data[qid])):
                    # exclude the current answer
                    gold_answers = self.gold_data[qid][0:i] + self.gold_data[qid][i + 1:]
                    em_sum += max(CJRCEvaluator.compute_exact(a, self.gold_data[qid][i]) for a in gold_answers)
                    f1_sum += max(CJRCEvaluator.compute_f1(a, self.gold_data[qid][i]) for a in gold_answers)
            else:
                exit("Gold answers should be multiple: {}={}".format(qid, self.gold_data[qid]))
            exact_scores[qid] = em_sum / len(self.gold_data[qid])
            f1_scores[qid] = f1_sum / len(self.gold_data[qid])
        return exact_scores, f1_scores

    def human_performance(self):
        exact_scores, f1_scores = self.get_raw_scores_human()
        return self.get_total_scores(exact_scores, f1_scores)

    def model_performance(self, pred_data):
        exact_scores, f1_scores = self.get_raw_scores(pred_data)
        return self.get_total_scores(exact_scores, f1_scores)
        # return self.get_qa_types_score(exact_scores, f1_scores)

    def get_total_scores(self, exact_scores, f1_scores):
        em_total, f1_total, turn_count = 0, 0, 0
        scores = {}
        for qid in self.gold_data:
            em_total += exact_scores.get(qid, 0)
            f1_total += f1_scores.get(qid, 0)
            turn_count += 1
        scores["overall"] = {'em': round(em_total / max(1, turn_count) * 100, 1),
                             'f1': round(f1_total / max(1, turn_count) * 100, 1),
                             'qas': turn_count}
        return scores

    def get_qa_types_score(self, exact_scores, f1_scores):
        qa_types = {}
        em_total = 0.0
        f1_total = 0.0
        total_count = 0
        for qid in self.gold_data:
            type = get_example_qa_type(self.id_to_example[qid])
            em_total += exact_scores.get(qid, 0)
            f1_total += f1_scores.get(qid, 0)
            total_count += 1
            if type not in qa_types:
                qa_types[type] = {}
                qa_types[type]["em_total"] = exact_scores.get(qid, 0)
                qa_types[type]['f1_total'] = f1_scores.get(qid, 0)
                qa_types[type]['qa_count'] = 1
            else:
                qa_types[type]["em_total"] += exact_scores.get(qid, 0)
                qa_types[type]['f1_total'] += f1_scores.get(qid, 0)
                qa_types[type]['qa_count'] += 1
        scores = OrderedDict()
        for type in qa_types.keys():
            scores[type] = {
                "em": round(qa_types[type]["em_total"] / max(1, qa_types[type]['qa_count']) * 100, 1),
                "f1": round(qa_types[type]['f1_total'] / max(1, qa_types[type]['qa_count']) * 100, 1),
                "qas": qa_types[type]['qa_count']
            }
        scores["F1"] = round(f1_total / max(1, total_count) * 100, 1)
        return scores


def get_example_qa_type(example: CAILExample):
    if example.is_impossible:
        return TYPE_NO_ANSWER
    if example.is_yes_no:
        if example.answer_texts[0] == "YES":
            return TYPE_YES
        elif example.answer_texts[0] == "NO":
            return TYPE_NO
        else:
            raise Exception()
    if len(example.answer_texts) == 1:
        return TYPE_SINGLE_SPAN
    if len(example.answer_texts) > 1:
        return TYPE_MULTI_SPAN
    raise Exception()

