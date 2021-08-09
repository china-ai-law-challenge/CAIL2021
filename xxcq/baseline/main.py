import os
from utils.utils_file import read_json_to_txt,save_result_json,creat_label

input_path = './datasets/CAIL司法文本信息抽取'
output_path = './output'

train_json = "你的训练集"
dev_json = "你的验证集"
test_json = "你的测试集"


train_set = "train.txt"
dev_set = "dev.txt"
test_set = "test.txt"

test_predictions = "test_predictions.txt"
test_submit = "test_predictions.json"


def main():
    # json to txt
    read_json_to_txt(os.path.join(input_path,train_json),
                     os.path.join(input_path, train_set))
    read_json_to_txt(os.path.join(input_path,dev_json),
                     os.path.join(input_path, dev_set))
    read_json_to_txt(os.path.join(input_path,test_json),
                     os.path.join(input_path, test_set))
    creat_label(input_path)

    # train/predict
    os.system("sh main.sh")

    # txt to json
    save_result_json(os.path.join(output_path,test_predictions),
                     os.path.join(output_path, test_submit),
                     os.path.join(input_path, test_json),
                     )

if __name__ == '__main__':
    main()
