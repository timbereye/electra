import json
import collections
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

retro_prediction_file = './data/retro_preds.json'


