import json
from copy import deepcopy

train = json.load(open('train-v2.0.json', 'r', encoding='utf-8'))
meta = json.load(open('answer_refine.meta', 'r', encoding='utf-8'))
for article in train['data']:
    for p in article['paragraphs']:
        new_p = deepcopy(p)
        for i, qa in enumerate(new_p['qas']):
            qid = qa['id']
            if qa['is_impossible']:
                p['qas'][i] = []
            else:
                p['qas'][i]['refine_class'] = meta[qid]['class']
        p['qas'] = list(filter(lambda x: len(x), p['qas']))

json.dump(train, open('train.json', 'w', encoding='utf-8'))
