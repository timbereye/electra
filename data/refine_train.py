import json

train = json.load(open('train-v2.0.json', 'r', encoding='utf-8'))
meta = json.load(open('answer_refine.meta', 'r', encoding='utf-8'))

for article in train['data']:
    for p in article['paragraphs']:
        for qa in p['qas']:
            qid = qa['id']
            refine_class = meta[qid]
            qa['refine_class'] = meta[qid]['class']

json.dump(train, open('train.json', 'w', encoding='utf-8'))
