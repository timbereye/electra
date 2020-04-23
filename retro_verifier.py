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


print(1)
