import argparse
import json
import os
import random
from gensim.summarization import bm25
import jieba
import numpy as np

parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--input', type=str, help='input path of the dataset directory.')
parser.add_argument('--output', type=str, help='output path of the prediction file.')

args = parser.parse_args()
input_path = args.input
input_query_path = os.path.join(input_path, 'query.json')
input_candidate_path = os.path.join(input_path, 'candidates')
output_path = args.output

if __name__ == "__main__":
    print('begin...')
    result = {}
    with open(os.path.join(os.path.dirname(__file__), 'stopword.txt'), 'r') as g:
        words = g.readlines()
    stopwords = [i.strip() for i in words]
    stopwords.extend(['.','（','）','-'])

    corpus = []
    lines = open(input_query_path, 'r').readlines()
    qs = [str(eval(line)['ridx']) for line in lines]
    # qs = os.listdir(input_candidate_path)
    for line in lines:
        query = str(eval(line)['ridx'])
        # model init
        result[query] = []
        files = os.listdir(os.path.join(input_candidate_path, query))
        for file_ in files:
            file_json = json.load(open(os.path.join(input_candidate_path, query, file_), 'r'))
            a = jieba.cut(file_json['ajjbqk'], cut_all=False)
            tem = " ".join(a).split()
            corpus.append([i for i in tem if not i in stopwords])
    bm25Model = bm25.BM25(corpus)
    
    for line in lines[:]:
        query = str(eval(line)['ridx'])
        a = jieba.cut(eval(line)['q'], cut_all=False)
        tem = " ".join(a).split()
        q = [i for i in tem if not i in stopwords]
        raw_rank_index = np.array(bm25Model.get_scores(q)).argsort().tolist()[::-1]
        rank_index = [i for i in raw_rank_index if i in range(qs.index(query)*100,(qs.index(query)+1)*100)]
        files = os.listdir(os.path.join(input_candidate_path, query))
        result[query] = [int(files[i%100].split('.')[0]) for i in rank_index]
    
    json.dump(result, open(os.path.join(output_path, 'prediction.json'), "w", encoding="utf8"), indent=2, ensure_ascii=False, sort_keys=True)
    print('ouput done.')
