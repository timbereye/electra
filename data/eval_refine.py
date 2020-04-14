import pickle
import json
from functional import seq
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from finetune.qa.qa_metrics import _compute_softmax
from data.eval import compute_f1
import random


def eval_class():
    refine = pickle.load(open('refine.pkl', 'rb'))

    y_true = []
    y_pred = []
    for qid, v in refine.items():
        y_true.append(v['true_label'])

        logits = v['refine_logits']
        sum_logits = np.sum(np.stack(logits, axis=-1), axis=-1)
        probs = _compute_softmax(sum_logits.tolist())

        pred = np.argmax(probs)
        pred = pred if probs[pred] > 0.8 else 0
        y_pred.append(pred)

    print(classification_report(y_true, y_pred))


def eval_preds():
    squad_preds = json.load(open('squad_preds.json', 'r', encoding='utf-8'))
    best_th = json.load(open('squad_eval.json', 'r', encoding='utf-8'))['best_exact_thresh']
    print("init eval:")
    xargs = f"python eval.py dev-v2.0.json squad_preds.json " \
            f"--na-prob-file squad_null_odds.json "
    os.system(xargs)

    all_nbest = pickle.load(open('all_nbest_file.pkl', 'rb'))
    refine_score = pickle.load(open('dev_refine.pkl', 'rb'))
    dev = json.load(open('dev-v2.0.json', 'r', encoding='utf-8'))['data']

    for article in dev:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                true_answers = qa['answers']
                if not true_answers:
                    true_answers = [{'text': ''}]
                v = refine_score[qid]
                probs = _compute_softmax(np.sum(np.stack(v['refine_logits'], axis=-1), axis=-1).tolist())
                refine_class = np.argmax(probs)
                refine_class = refine_class if probs[refine_class] > 0.9 else 0
                nbest = sorted(all_nbest[qid], key=lambda x: x['probability'], reverse=True)
                # if refine_class == 1:
                #     select_answers = (seq(nbest)
                #                       .filter(lambda x: squad_preds[qid] in x['text'])
                #                       .sorted(lambda x: x['probability'], reverse=True)
                #                       ).list()
                #
                # elif refine_class == 2:
                #     select_answers = (seq(nbest)
                #                       .filter(lambda x: x['text'] in squad_preds[qid])
                #                       .sorted(lambda x: x['probability'], reverse=True)
                #                       ).list()
                # else:
                #     continue
                # if len(select_answers) < 2:
                #     continue

                # text = select_answers[1]['text']
                text = list(seq(nbest[:5])
                            .map(lambda x: {max([compute_f1(a['text'], x['text']) for a in true_answers]): x['text']})
                            .sorted(lambda x: list(x.keys())[0], reverse=True)
                            .list()[0].values())[0]
                if random.random() > 0:
                    squad_preds[qid] = text
                else:
                    squad_preds[qid] = nbest[random.randint(0, 1)]['text']
    json.dump(squad_preds, open('squad_refine_preds.json', 'w', encoding='utf-8'))

    print("refine eval:")
    xargs = f"python eval.py dev-v2.0.json squad_refine_preds.json " \
            f"--na-prob-file squad_null_odds.json"
    os.system(xargs)


eval_preds()
