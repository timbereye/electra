import json
import random

train = json.load(open('train-v2.0.json', 'r', encoding='utf-8'))
meta = json.load(open('answer_refine.meta', 'r', encoding='utf-8'))

for article in train['data']:
    for p in article['paragraphs']:
        for qa in p['qas']:
            qid = qa['id']
            qa['refine_class'] = meta[qid]['class']
            if qa['is_impossible']:
                p['qas'].remove(qa)

json.dump(train, open('train.json', 'w', encoding='utf-8'))
