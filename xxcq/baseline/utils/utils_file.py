import json
import os


class Seqence:
    def __init__(self,tokens,id):
        self.context = None
        self.tags = None
        self.id = id
        self.tokens = tokens

    def extract_token_and_tag(self):
        context = [t.word for t in self.tokens]

        label = "O"
        tags = []
        for token in self.tokens:

            if True in token.__dict__.values():
                for k,v in token.__dict__.items():
                    if k and v and not k == "word":
                        if label[0] == "O":
                            label = "B-" + k
                        elif label[0] == "B":
                            if label[2:] == k:
                                label = "I-" + k
                            else:
                                label = "B-" + k
                        elif label[0] == "I":
                            if label[2:] == k:
                                label = "I-" + k
                            else:
                                label = "B-" + k
                        break
            else:
                label = "O"
            tags.append(label)

        self.context = context
        self.tags = tags


class Token:
    def __init__(self,word):
        if not word == "�":
            self.word = word
        else:
            self.word = "*"
        self.NHCS = None
        self.NHVI = None
        self.NCSM = None
        self.NCGV = None
        self.NASI = None
        self.NT = None
        self.NS = None
        self.NO = None
        self.NATS = None
        self.NCSP = None

def read_json_to_txt(input,output):
    samples = []
    with open(input,"r",encoding="utf-8") as f:
        for l in f.readlines():
            line = json.loads(l.strip())
            if "entities" in line.keys():
                context,entities,id = line["context"],line["entities"],line["id"]
            else:
                context,id = line["context"],line["id"]
                entities = []
            tokens = [Token(word) for word in context]

            for entity in entities:
                label,spans = entity["label"],entity["span"]
                for span in spans:
                    span = span.split(";")
                    for p in range(int(span[0]),int(span[1])):
                        if label == "NHCS":
                            tokens[p].NHCS = True
                        elif label == "NHVI":
                            tokens[p].NHVI = True
                        elif label == "NCSM":
                            tokens[p].NCSM = True
                        elif label == "NCGV":
                            tokens[p].NCGV = True
                        elif label == "NASI":
                            tokens[p].NASI = True
                        elif label == "NT":
                            tokens[p].NT = True
                        elif label == "NS":
                            tokens[p].NS = True
                        elif label == "NO":
                            tokens[p].NO = True
                        elif label == "NATS":
                            tokens[p].NATS = True
                        elif label == "NCSP":
                            tokens[p].NCSP = True


            sample = Seqence(tokens,id)
            sample.extract_token_and_tag()

            assert len(sample.context) == len(sample.tags)

            samples.append(sample)

    with open(output,"w",encoding="utf-8") as f:
        for sample in samples:
            for w,t in zip(sample.context,sample.tags):
                f.write(w+" "+t+"\n")
            f.write("\n")

def creat_label(path):
    labels = []
    with open(os.path.join(path,"train.txt"),"r",) as f:
        for l in f.readlines():
            if not l.startswith("\n"):
                label = l.strip().split()[1]
                if label not in labels:
                    labels.append(label)
    with open(os.path.join(path,"labels.txt"),"w") as f:
        for label in labels:
            f.write(label + "\n")

def check(path):
    label = ""
    with open(path,"r",encoding="utf-8") as f:
        for i,l in enumerate(f.readlines()):
            if not l.startswith("\n"):
                tag = l.split()[1]
                if label == "B" and tag[0] == "B":
                    print(i)

def get_position(tag:list,tag_name):
    spans = []
    start = 0
    end = 0
    flag = False
    for i,t in enumerate(tag):
        if t and not flag:
            start = i
            flag = True
        elif t and flag:
            end = i
            flag = True
        elif not t and flag:
            end = i
            flag = False
            spans.append(str(start)+";"+str(end))
        else:
            pass
    return {"label":tag_name,"span":spans}

def save_result_json(input_json_path,output_json_path,test_json_path):
    seqences = []
    with open(input_json_path,"r") as f:
        seqence = []
        for l in f.readlines():
            if not l.startswith("\n"):
                line = l.split()
                word,label = line[0],line[1]

                token = Token(word)
                if "NHCS" in label:
                    token.NHCS = True
                elif "NHVI" in label:
                    token.NHVI = True
                elif "NCSM" in label:
                    token.NCSM = True
                elif "NCGV" in label:
                    token.NCGV = True
                elif "NASI" in label:
                    token.NASI = True
                elif "NT" in label:
                    token.NT = True
                elif "NS" in label:
                    token.NS = True
                elif "NO" in label:
                    token.NO = True
                elif "NATS" in label:
                    token.NATS = True
                elif "NCSP" in label:
                    token.NCSP = True
                seqence.append(token)
            else:
                if len(seqence) > 0:
                    seqences.append(seqence)
                    seqence = []

    samples = []
    with open(test_json_path,"r",encoding="utf-8") as f:
        for test_data, seqence in zip(f.readlines(), seqences):
            test_data = json.loads(test_data)
            id = test_data["id"]

            context = [token.word for token in seqence]

            entity = []
            tag_NHCS = [token.NHCS for token in seqence]
            entity.append(get_position(tag_NHCS, "NHCS"))
            tag_NHVI = [token.NHVI for token in seqence]
            entity.append(get_position(tag_NHVI, "NHVI"))
            tag_NCSM = [token.NCSM for token in seqence]
            entity.append(get_position(tag_NCSM, "NCSM"))
            tag_NCGV = [token.NCGV for token in seqence]
            entity.append(get_position(tag_NCGV, "NCGV"))
            tag_NASI = [token.NASI for token in seqence]
            entity.append(get_position(tag_NASI, "NASI"))
            tag_NT = [token.NT for token in seqence]
            entity.append(get_position(tag_NT, "NT"))
            tag_NS = [token.NS for token in seqence]
            entity.append(get_position(tag_NS, "NS"))
            tag_NO = [token.NO for token in seqence]
            entity.append(get_position(tag_NO, "NO"))
            tag_NATS = [token.NATS for token in seqence]
            entity.append(get_position(tag_NATS, "NATS"))
            tag_NCSP = [token.NCSP for token in seqence]
            entity.append(get_position(tag_NCSP, "NCSP"))

            sample = {"id": id, "context": "".join(context), "entities": entity}
            samples.append(sample)



    with open(output_json_path,"w",) as f:
        for sample in samples:
            f.write(json.dumps(sample))
            f.write("\n")


def count_entity_number(path):
    samples = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            samples.append(data)

    counter = 0
    for sample in samples:
        for entity in sample["entities"]:
            for e in entity["span"]:
                counter += 1
    print("entity_counter: ",counter)

def get_score(ground_truth_path, output_path):

    try:
        ground_truth = {}
        prediction = {}
        with open(ground_truth_path, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                id = data['id']

                ground_truth[id] = data
        with open(output_path, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                id = data['id']
                data.pop('id')
                prediction[id] = data

        ground_truth_num = 2987

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
        print(e)
        return {"p": -1, "r": -1, "f": -1}

if __name__ == '__main__':
    # main("./xxcq_small.json","./train.txt")
    # main("./xxcq_test.json","./test.txt")
    # main("./xxcq_test.json","./dev.txt")
    # creat_label()
    # check("./test.txt")
    # save_result_json("./test.txt","./test.json")
    # print(get_score("xxcq_test.json","test.json"))
    # count_entity_number("xxcq_test.json")
    pass
