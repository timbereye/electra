import json
import pickle
from copy import deepcopy
from functional import seq
from finetune.qa.squad_official_eval import compute_f1

train = json.load(open('train-v2.0.json', 'r', encoding='utf-8'))
train_all_nbest = pickle.load(open('train_all_nbest_file.pkl', 'rb'))
count = 0

for article in train['data']:
    for p in article['paragraphs']:
        del p['context']
        new_qas = []
        for qa in p['qas']:
            qid = qa['id']
            gold_answers = qa['answers']
            if not gold_answers:
                continue
            nbest = train_all_nbest[qid][:5]
            new_qa = []
            for i, nb in enumerate(nbest):
                pred = nb['text']
                a = qa['answers'][0]['text']
                f1 = compute_f1(a, pred)
                new_qa.append({"f1_score": f1,
                               "pred_answer": pred,
                               "question": qa['question'],
                               "id": f"{qid}_{i}"})
            new_qas.extend(new_qa)
        p['qas'] = new_qas
        count += len(new_qas)

print(count)

json.dump(train, open('train.json', 'w', encoding='utf-8'))
