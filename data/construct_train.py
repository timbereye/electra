import json

train = json.load(open('train-v2.0.json', 'r', encoding='utf-8'))

train_preds = json.load(open('train_squad_preds.json', 'r', encoding='utf-8'))
train_null_odds = json.load(open('train_squad_null_odds.json', 'r', encoding='utf-8'))
train_eval = json.load(open('train_squad_eval.json', 'r', encoding='utf-8'))

th = train_eval['best_exact_thresh']

for article in train['data']:
    for p in article['paragraphs']:
        for qa in p['qas']:
            qid = qa['id']
            is_impossible = qa['is_impossible']
            if is_impossible and train_null_odds[qid] < th:
                qa['is_impossible'] = 2

json.dump(train, open('train.json', 'w', encoding='utf-8'))
