import sys
import json

def get_score(ground_truth_path, output_path): 
    try:
        ground_truth = {}
        prediction = {}
        with open(ground_truth_path, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                id = data['id']
                data.pop('id')        
                ground_truth[id] = data
        with open(output_path, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                id = data['id']
                data.pop('id')
                prediction[id] = data
        
        ground_truth_num = 300 # As example
        prediction_num = 0
        tp = 0

        for id, ground_truth_data in ground_truth.items():
            try:
                pred_data = prediction[id]
            except KeyError:
                continue
            ground_truth_entities_dict = {}
            for entitie in ground_truth_data['entities']:
                ground_truth_entities_dict[entitie['label']] = entitie['span']
            pred_entities_dict = {}
            for entitie in pred_data['entities']:
                prediction_num += len(entitie['span'])
                pred_entities_dict[entitie['label']] = entitie['span']
            for label in ground_truth_entities_dict.keys():
                tp += len(set(ground_truth_entities_dict[label]).intersection(set(pred_entities_dict[label])))

        p = tp / prediction_num
        r = tp / ground_truth_num
        f = 2 * p * r / ( p + r )
            
        s1 = round(p * 100, 2)
        s2 = round(r * 100, 2)
        s3 = round(f * 100, 2)
        return {"p": s1, "r": s2, "f": s3}
    except Exception as e:
        return {"p": -1, "r": -1, "f": -1}

if __name__ == '__main__':
    get_score(sys.argv[1], sys.argv[2])
