# -*- encoding: utf-8 -*-

"""
用途      ：BERT预测模块
@File    :bert_predict.py
@Time    :2020/11/9 14:27
@Author  :hyyd
@Software: PyCharm
"""
import sys

sys.path.extend(['.', '..', ])

import tensorflow as tf
from step1_model.bert import tokenization, modeling
from utils import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def text_process(text1, tokenizer, max_seq_length):
    token_1 = tokenizer.tokenize(text1)
    truncate_length = max_seq_length - 2
    token_1 = token_1[:truncate_length]
    tokens, segment_ids = [], []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in token_1:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return input_ids, input_mask, segment_ids


def init(max_sequence_length, bert_config_file, model_path, vocab_file, n_class):
    sess = tf.Session()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    input_ids = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name='segment_ids')

    with sess.as_default():
        model = modeling.BertModel(config=bert_config, is_training=False, input_ids=input_ids,
                                   input_mask=input_mask, token_type_ids=segment_ids, use_one_hot_embeddings=False)

        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable("output_weights", [n_class, hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable("output_bias", [n_class], initializer=tf.zeros_initializer())

        with tf.variable_scope('loss'):
            logit = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logit, output_bias)
            probabilities = tf.nn.sigmoid(logits)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)

    return sess, tokenizer


def predict(sess, input_ids, input_mask, segment_ids):
    input_ids_tensor = sess.graph.get_tensor_by_name('input_ids:0')
    input_mask_tensor = sess.graph.get_tensor_by_name('input_mask:0')
    segment_ids_tensor = sess.graph.get_tensor_by_name('segment_ids:0')
    output_tensor = sess.graph.get_tensor_by_name('loss/Sigmoid:0')

    fd = {input_ids_tensor: [input_ids], input_mask_tensor: [input_mask], segment_ids_tensor: [segment_ids]}
    output_result = sess.run([output_tensor], feed_dict=fd)
    return output_result


def create_prediction_model(vocab_file='./step1_model/BERT_BASE_DIR/vocab.txt',
                            bert_config_file='./step1_model/BERT_BASE_DIR/bert_config.json',
                            ckpt_dir='./step1_model/output',
                            max_seq_len=500,
                            n_classes=133):
    bert_model = BERTPredictModule(vocab_file, bert_config_file, ckpt_dir, max_seq_len, n_classes)
    return bert_model


def single_file_predict_pipeline(text, bert_model):
    """
    输入整篇文书，输出分句预测结果
    :param text:
    :param bert_model:
    :return:
    """
    sentence_list = split_to_sentence(remove_blank(text))

    prediction_list = []
    for sent in sentence_list:
        prediction_answer = bert_model.sentence_inference(sent)
        prediction_list.append(prediction_answer)
    return sentence_list, prediction_list


class BERTPredictModule(object):
    _isinstance = False

    def __init__(self, vocab_file, bert_config_file, ckpt_dir, max_seq_len, n_classes):
        # BERT相关配置
        self.vocab_file = vocab_file
        self.bert_config_file = bert_config_file

        # 训练好的BERT模型
        self.model_path = tf.train.latest_checkpoint(ckpt_dir)
        self.max_sequence_length = max_seq_len
        self.n_class = n_classes

        self.sess, self.tokenizer = init(self.max_sequence_length, self.bert_config_file, self.model_path,
                                         self.vocab_file, self.n_class)

    def inference(self, input_data):
        result_list = []
        for text1 in input_data:
            input_ids, input_mask, segment_ids = text_process(text1, self.tokenizer, self.max_sequence_length)
            result = predict(self.sess, input_ids, input_mask, segment_ids)
            result_list.append(result[0][0])
        return result_list

    def sentence_inference(self, sentence):
        input_ids, input_mask, segment_ids = text_process(sentence, self.tokenizer, self.max_sequence_length)
        result = predict(self.sess, input_ids, input_mask, segment_ids)
        return result[0][0]

    def __new__(cls, *args, **kwargs):
        if cls._isinstance:
            return cls._isinstance
        cls._isinstance = object.__new__(cls)
        return cls._isinstance
