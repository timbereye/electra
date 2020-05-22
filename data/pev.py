import json
from copy import deepcopy
import os
import pickle
import numpy as np
from functional import seq


preds_file = 'atrlp8876_squad_preds.json'
null_odds_file = 'atrlp8876_squad_null_odds.json'
eval_file = 'atrlp8876_squad_eval.json'
pv_null_odds_file = 'atrlp8876_pv_squad_null_odds.json'
answer_candidates_score_file = 'dev_f1_predict_results2.pkl'
dev_all_nbest_file = 'atrlp8876_dev_all_nbest.pkl'
tmp_eval_file = 'tmp_eval.json'
tmp_preds_file = 'tmp_preds.json'
tmp_null_odds_file = 'tmp_null_odds.json'

preds = json.load(open(preds_file, 'r', encoding='utf-8'))
null_odds = json.load(open(null_odds_file, 'r', encoding='utf-8'))
best_th = json.load(open(eval_file, 'r', encoding='utf-8'))["best_exact_thresh"]
pv_null_odds = json.load(open(pv_null_odds_file, 'r', encoding='utf-8'))
answer_candidates_score = pickle.load(open(answer_candidates_score_file, 'rb'))
dev_all_nbest = pickle.load(open(dev_all_nbest_file, 'rb'))

# length = 5
# believe_f1_th = 0.7
# believe_prob_th = 0.2
# for qid in preds:
#     chooses = (seq(range(length))
#                .map(lambda x: answer_candidates_score.get(f"{qid}_{x}", None))
#                .map(lambda x: x if x is None else {'f1_pred': np.max([y['predictions'] for y in x])})
#                .zip(dev_all_nbest[qid][:length])
#                .filter(lambda x: x[0])
#                .map(lambda x: {'text': x[1]['text'],
#                                'f1_pred': x[0]['f1_pred'],
#                                'prob': x[1]['probability']})
#                .sorted(lambda x: x['f1_pred'], reverse=True)
#                ).list()
#     if len(chooses):
#         max_prob = seq(chooses).map(lambda x: x['prob']).max()
#         if chooses[0]['f1_pred'] > believe_f1_th and max_prob - chooses[0]['prob'] < believe_prob_th:
#             preds[qid] = chooses[0]['text']

merge_null_odds = deepcopy(null_odds)

for k, v in merge_null_odds.items():
    merge_null_odds[k] = null_odds[k] + pv_null_odds[k]
json.dump(merge_null_odds, open(tmp_null_odds_file, 'w', encoding='utf-8'))

xargs = f"python eval.py dev-v2.0.json {preds_file} --na-prob-file {tmp_null_odds_file} --out-file {tmp_eval_file}"
os.system(xargs)

new_sh = json.load(open(tmp_eval_file, 'r', encoding='utf-8'))["best_exact_thresh"]

for k, v in merge_null_odds.items():
    if v > new_sh:
        preds[k] = ""
json.dump(preds, open(tmp_preds_file, 'w', encoding='utf-8'))

print("pv eval:")
xargs = f"python eval.py dev-v2.0.json {tmp_preds_file}"
os.system(xargs)
