import json
from copy import deepcopy
import os
import pickle
from matplotlib import pyplot as plt
from data.eval import get_raw_scores
import numpy as np
import time
from functional import seq
from data.eval import normalize_answer

preds = json.load(open('squad_preds.json', 'r', encoding='utf-8'))
null_odds = json.load(open('squad_null_odds.json', 'r', encoding='utf-8'))
answer_null_odds = json.load(open('squad_null_odds_answer_model.json', 'r', encoding='utf-8'))

best_th = json.load(open('squad_eval.json', 'r', encoding='utf-8'))["best_exact_thresh"]

print("init eval:")
xargs = f"python eval.py dev-v2.0.json squad_preds.json --na-prob-file squad_null_odds.json "
os.system(xargs)

answer_chooses = pickle.load(open('dev_f1_predict_results3.pkl', 'rb'))
nbest = pickle.load(open('dev_all_nbest_file.pkl', 'rb'))

tmp_file = 'tmp_preds'
tmp_eval_file = 'tmp_eval_file'
final_preds_file = 'final_preds.json'

length = 5
believe_f1_th = 0.7
believe_prob_th = 0.2

for qid in preds:
    chooses = (seq(range(length))
               .map(lambda x: answer_chooses.get(f"{qid}_{x}", None))
               .map(lambda x: x if x is None else {'f1_pred': np.max([y['predictions'] for y in x])})
               .zip(nbest[qid][:length])
               .filter(lambda x: x[0])
               .map(lambda x: {'text': x[1]['text'],
                               'f1_pred': x[0]['f1_pred'],
                               'prob': x[1]['probability']})
               .sorted(lambda x: x['f1_pred'], reverse=True)
               ).list()
    if len(chooses):
        max_prob = seq(chooses).map(lambda x: x['prob']).max()
        if chooses[0]['f1_pred'] > believe_f1_th and max_prob - chooses[0]['prob'] < believe_prob_th:
            preds[qid] = normalize_answer(chooses[0]['text'])

json.dump(preds, open(final_preds_file, 'w', encoding='utf-8'))

print("final eval:")
xargs = f"python eval.py dev-v2.0.json preds_by_f1.json --na-prob-file squad_null_odds.json " \
        f"--out-file {tmp_eval_file}"
os.system(xargs)

for k, v in null_odds.items():
    if v > best_th:
        preds[k] = ""

json.dump(preds, open(tmp_file, 'w', encoding='utf-8'))
xargs = f"python eval.py dev-v2.0.json {tmp_file} " \
        f"--na-prob-file squad_null_odds_answer_model.json --out-file {tmp_eval_file}"
os.system(xargs)
new_sh = json.load(open(tmp_eval_file, 'r', encoding='utf-8'))["best_exact_thresh"]
for k, v in answer_null_odds.items():
    if v > new_sh:
        preds[k] = ""

json.dump(preds, open(final_preds_file, 'w', encoding='utf-8'))
xargs = f"python eval.py dev-v2.0.json {final_preds_file} "
os.system(xargs)
