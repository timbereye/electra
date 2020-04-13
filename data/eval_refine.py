import pickle
import json
from functional import seq
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from finetune.qa.qa_metrics import _compute_softmax

refine = pickle.load(open('refine.pkl', 'rb'))

y_true = []
y_pred = []
for qid, v in refine.items():
    y_true.append(v['true_label'])

    logits = v['refine_logits']
    sum_logits = np.sum(np.stack(logits, axis=-1), axis=-1)
    probs = _compute_softmax(sum_logits.tolist())

    y_pred.append(np.argmax(probs))

print(classification_report(y_true, y_pred))
