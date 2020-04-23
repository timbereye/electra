import json
import pickle
import collections
from copy import deepcopy
import os
from matplotlib import pyplot as plt
from data.eval import get_raw_scores
import numpy as np
import time

preds = json.load(open('./data/squad_preds.json', 'r', encoding='utf-8'))
null_odds = json.load(open('./data/squad_null_odds.json', 'r', encoding='utf-8'))
all_nbest = pickle.load(open('./data/dev_all_nbest.pkl', 'rb'))
all_logits = pickle.load(open('./data/dev_all_logits.pkl', 'rb'))
answer_null_odds = json.load(open('./data/squad_null_odds_answer_model.json', 'r', encoding='utf-8'))
dev = json.load(open('./data/dev-v2.0.json', 'r', encoding='utf-8'))['data']

retro_prediction_file = './data/retro_preds.json'

score_has = collections.OrderedDict()
score_na = collections.OrderedDict()


def get_na_prob(qid):
    logits = all_logits[qid]
    start_na_prob = []
    for logit in logits:
        start_na_prob.append(np.exp(logit['start_logits']))

    start_na_prob = np.mean(start_na_prob)

    odd_prob = 0.5 * 1 / (1 + np.exp(-null_odds[qid])) + 0.5 * 1 / (1 + np.exp(-answer_null_odds[qid]))

    return start_na_prob + odd_prob


for qid in preds:
    score_has[qid] = np.exp(all_nbest[qid][0]['start_logit']) + np.exp(all_nbest[qid][0]['end_logit'])
    score_na[qid] = get_na_prob(qid)

for qid in preds:
    if score_na[qid] - score_has[qid] > 0.7:
        preds[qid] = ""

json.dump(preds, open(retro_prediction_file, 'w', encoding='utf-8'))
xargs = f"python ./data/eval.py ./data/dev-v2.0.json {retro_prediction_file}"
os.system(xargs)
