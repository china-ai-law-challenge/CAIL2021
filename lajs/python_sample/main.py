import argparse
import json
import os
import random

parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--input', type=str, help='input path of the dataset directory.')
parser.add_argument('--output', type=str, help='output path of the prediction file.')

args = parser.parse_args()
input_path = args.input
input_query_path = os.path.join(input_path, 'query.json')
input_candidate_path = os.path.join(input_path, 'candidates')
output_path = args.output

if __name__ == "__main__":
    result = {}
    qs = os.listdir(input_candidate_path)
    for query in qs:
        result[query] = []
        files = os.listdir(os.path.join(input_candidate_path, query))
        for file_ in files:
            if len(result[query]) == 30:
                break
            else:
                if random.randint(1, 3) > 1:
                    file_idx = int(file_.split('.')[0])
                    result[query].append(file_idx)
    
    json.dump(result, open(os.path.join(output_path, 'prediction.json'), "w", encoding="utf8"), indent=2, ensure_ascii=False, sort_keys=True)
