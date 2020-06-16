# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:52
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import json
from copy import deepcopy
import random
import uuid
from textattack.augmentation import CharSwapAugmenter

train = json.load(open('train_aug1w.json', 'r', encoding='utf-8'))
augmenter = CharSwapAugmenter()

qid_qas = {}

for article in train['data']:
    for paragraph in article["paragraphs"]:
        for qa in paragraph['qas']:
            qid = qa['id']
            qid_qas[qid] = qa

char_swap_qid_qas = deepcopy(qid_qas)
for qid, qa in char_swap_qid_qas:
    if not qa['is_impossible']:
        question = qa['question']
        char_swap_question = augmenter.augment(question)[0]
        qa['question'] = char_swap_question
        qa['is_impossible'] = True
        qa['answers'] = []
        qa['char_swap'] = True

first_half_qid_qas = deepcopy(qid_qas)
for qid, qa in first_half_qid_qas:
    if not qa['is_impossible']:
        question = qa['question']
        first_half_question = question[:len(question) // 2]
        qa['question'] = first_half_question
        qa['is_impossible'] = True
        qa['answers'] = []
        qa['first_half'] = True

swap_count = 0
half_count = 0
for article in train['data']:
    for paragraph in article["paragraphs"]:
        new_qas = deepcopy(paragraph['qas'])
        for qa in paragraph['qas']:
            qid = qa['id']
            if char_swap_qid_qas[qid].get('char_swap', False) and random.random() < 0.15 and swap_count < 10000:
                qa = char_swap_qid_qas[qid]
                qa.pop('char_swap')
                qa['id'] = uuid.uuid1().__str__()
                new_qas.append(qa)
                swap_count += 1
            if first_half_qid_qas[qid].get('first_half', False) and random.random() < 0.15 and half_count < 10000:
                qa = first_half_qid_qas[qid]
                qa.pop('first_half')
                qa['id'] = uuid.uuid1().__str__()
                new_qas.append(qa)
                half_count += 1
        paragraph['qas'] = new_qas

json.dump(train, open('train.json', 'w', encoding='utf-8'))
