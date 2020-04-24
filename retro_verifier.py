import json
import pickle
import collections
from functional import seq
from copy import deepcopy
import os
from matplotlib import pyplot as plt
from data.eval import get_raw_scores, make_qid_to_has_ans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import numpy as np
import time

preds = json.load(open('./data/squad_preds.json', 'r', encoding='utf-8'))
null_odds = json.load(open('./data/squad_null_odds.json', 'r', encoding='utf-8'))
all_nbest = pickle.load(open('./data/dev_all_nbest.pkl', 'rb'))
answer_null_odds = json.load(open('./data/squad_null_odds_answer_model.json', 'r', encoding='utf-8'))
dev = json.load(open('./data/dev-v2.0.json', 'r', encoding='utf-8'))['data']

retro_prediction_file = './data/retro_preds.json'
score_diff = collections.OrderedDict()


def get_probs(qid):
    best = all_nbest[qid][0]
    score_has = np.exp(best['start_logit']) + np.exp(best['start_logit'])
    score_na = best['start_cls_logits'] + best['end_cls_logits']

    y_hat = 1 / (1 + np.exp(-answer_null_odds[qid]))
    y_ba = 1 / (1 + np.exp(-null_odds[qid]))

    return [score_has, score_na, y_hat, y_ba]


for th in np.arange(-1, 0.5, 0.1):
    for qid in preds:
        scores = get_probs(qid)
        score_diff[qid] = scores[0] - (0.5 * scores[1] + 0.5 * (0.5 * scores[2] + 0.5 * scores[3]))
        if score_diff[qid] < th:
            preds[qid] = ""

    print(th)
    json.dump(preds, open(retro_prediction_file, 'w', encoding='utf-8'))
    xargs = f"python ./data/eval.py ./data/dev-v2.0.json {retro_prediction_file} "
    os.system(xargs)
