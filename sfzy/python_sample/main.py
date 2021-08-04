import json
from random import choice
input_path = '/input/input.json'  # input file path
output_path = '/output/result.json'  # output file path
if __name__ == "__main__":
    with open(output_path, 'w', encoding='utf8') as fw:
        with open(input_path, 'r', encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                id = data.get('id')
                answer = choice(data.get('candidates'))
                rst = dict(
                    id=id,
                    answer=answer
                )
                fw.write(json.dumps(rst, ensure_ascii=False) + '\n')
