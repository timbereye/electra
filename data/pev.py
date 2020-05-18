import json
from copy import deepcopy
import os
import pickle
import numpy as np
import time
from functional import seq

preds_file = 'squad_preds.json'
null_odds_file = 'squad_null_odds.json'
eval_file = 'squad_eval.json'
pv_null_odds_file = 'pv_squad_null_odds.json'
tmp_eval_file = 'tmp_eval.json'
tmp_preds_file = 'tmp_preds.json'
tmp_null_odds_file = 'tmp_null_odds.json'

preds = json.load(open(preds_file, 'r', encoding='utf-8'))
null_odds = json.load(open(null_odds_file, 'r', encoding='utf-8'))
best_th = json.load(open(eval_file, 'r', encoding='utf-8'))["best_exact_thresh"]
pv_null_odds = json.load(open(pv_null_odds_file, 'r', encoding='utf-8'))

merge_null_odds = deepcopy(null_odds)

for k, v in merge_null_odds.items():
    merge_null_odds[k] += pv_null_odds[k]
json.dump(merge_null_odds, open(tmp_null_odds_file, 'w', encoding='utf-8'))

xargs = f"python eval.py dev-v2.0.json {preds_file} --na-prob-file {tmp_null_odds_file} "
os.system(xargs)
# print("init eval:")
# xargs = f"python eval.py dev-v2.0.json {preds_file} --na-prob-file {null_odds_file} --out-file {tmp_eval_file}"
# os.system(xargs)
#
# new_sh = json.load(open(tmp_eval_file, 'r', encoding='utf-8'))["best_exact_thresh"]
#
# for k, v in null_odds.items():
#     if v > new_sh:
#         preds[k] = ""
# json.dump(preds, open(tmp_preds_file, 'w', encoding='utf-8'))
#
# print("pv eval:")
# xargs = f"python eval.py dev-v2.0.json {tmp_preds_file} --na-prob-file {pv_null_odds_file}"
# os.system(xargs)
