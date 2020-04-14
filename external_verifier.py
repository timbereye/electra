import json
from copy import deepcopy
import os
from matplotlib import pyplot as plt
from data.eval import get_raw_scores
import numpy as np
import time

prediction_file = './data/squad_preds.json'
prediction = json.load(open(prediction_file, 'r', encoding='utf-8'))

null_odds_file = './data/squad_null_odds.json'
null_odds = json.load(open(null_odds_file, 'r', encoding='utf-8'))

eval_file = './data/squad_eval.json'
ts = json.load(open(eval_file, 'r', encoding='utf-8'))
best_th = ts["best_exact_thresh"]

answer_null_odds_file = './data/squad_null_odds_answer_model.json'
answer_null_odds = json.load(open(answer_null_odds_file, 'r', encoding='utf-8'))

dev_file = './data/dev-v2.0.json'
dev = json.load(open(dev_file, 'r', encoding='utf-8'))['data']

merge_prediction_file = './data/merge_preds.json'


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


qid_to_has_ans = make_qid_to_has_ans(json.load(open('./data/dev-v2.0.json', 'r', encoding='utf-8'))["data"])


def eval_twice():
    tmp_file = './data/tmp_preds.json'
    tmp_eval_file = './data/tmp_eval.json'
    final_preds_file = "./data/final_squad_preds.json"
    xargs = f"python ./data/eval.py ./data/dev-v2.0.json ./data/squad_preds.json " \
            f"--na-prob-file ./data/squad_null_odds.json --na-prob-thresh {best_th}"
    os.system(xargs)
    for k, v in null_odds.items():
        if v > best_th:
            prediction[k] = ""
    json.dump(prediction, open(tmp_file, 'w', encoding='utf-8'))
    xargs = f"python ./data/eval.py ./data/dev-v2.0.json {tmp_file} " \
            f"--na-prob-file ./data/squad_null_odds_answer_model.json --out-file {tmp_eval_file}"
    os.system(xargs)
    new_sh = json.load(open(tmp_eval_file, 'r', encoding='utf-8'))["best_exact_thresh"]
    for k, v in answer_null_odds.items():
        if v > new_sh:
            prediction[k] = ""
    json.dump(prediction, open(final_preds_file, 'w', encoding='utf-8'))
    xargs = f"python ./data/eval.py ./data/dev-v2.0.json {final_preds_file} "

    os.system(xargs)


# xargs = "python ./data/eval.py ./data/dev-v2.0.json ./data/squad_preds.json --na-prob-file ./data/squad_null_odds.json --na-prob-thresh -2.75"
# os.system(xargs)
#
# start = time.time()
# simple_replace_with_null_odds()
# xargs = "python ./data/eval.py ./data/dev-v2.0.json ./data/merge_preds.json"
# os.system(xargs)
# print(f"cost time: {time.time() - start}")
eval_twice()
