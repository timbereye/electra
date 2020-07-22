#!/usr/bin/env python
# encoding:utf-8
# -----------------------------------------#
# Filename:     meta_predict.py
#
# Description:
# Version:      1.0
# Created:      2020/7/22 14:24
# Author:       chenxiang@myhexin.com
# Company:      www.iwencai.com
#
# -----------------------------------------#
import math
import json
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import joblib
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, f1_score
# import itertools
# from sklearn.ensemble import StackingClassifier, RandomForestClassifier
# from xgboost import XGBClassifier


class DataLoader(object):
    def __init__(self, pred_files, labels_file=None):
        """
        Data loader class.
        Args:
            pred_files: {k: v} samples prediction
            labels_files: {k: v} samples label
        """
        self.pred_files = pred_files
        self.labels_file = labels_file

    def load(self, do_shuffle=False):
        K = len(self.pred_files)
        pred_all = []
        for i in range(K):
            with tf.gfile.Open(self.pred_files[i], "r") as fp:
                predictions = json.load(fp)
                pred_all.append(predictions)
        keys = sorted(list(pred_all[0].keys()))

        X = []
        for k in keys:
            x = []
            for i in range(K):
                x.append(pred_all[i][k])
            X.append(x)
        print("Samples Num: {}".format(len(keys)))

        if self.labels_file:
            with tf.gfile.Open(self.labels_file, "r") as fp:
                labels = json.load(fp)
                Y = []
                for k in keys:
                    Y.append(labels[k])
                tmp = [0] * K
                for i,x in enumerate(X):
                    for j, p in enumerate(x):
                        if int(p > 0) == Y[i]:
                            tmp[j] += 1
                tmp = [x/len(keys) for x in tmp]
                print("{} - Previous acc: {}".format(self.labels_file, tmp))
        else:
            Y = [0] * len(keys)

        if do_shuffle:
            X, Y = shuffle(X, Y)
        return np.array(X), np.array(Y), keys


def best_pred(model, X, keys):
    pred = model.predict_proba(X)[:, 1]
    ret = {}
    for i, key in enumerate(keys):
        ret[key] = float(pred[i])
    return ret


# 注意模型顺序
pv_odds_files_example = [
    "./data/bs32_seq384_lr5e-05_ep2.0_dev.json",
    "./data/bs32_seq512_lr3e-05_ep3_dev.json",
    "./data/bs32_seq512_lr5e-05_ep2.0_dev.json",
    "./data/albert_pv_2_384_2e-5_dev.json",
]


def predict(pv_odds_files=pv_odds_files_example, bagging_odds_file="data/bagging_odds.json",
            output_odds_file="data/final_odds.json", model_file="models/rf1.pkl"):
    model = joblib.load(model_file)
    data_loader = DataLoader(pv_odds_files)
    X, _, keys = data_loader.load()

    pred_odds = best_pred(model, X, keys)
    bagging_odds = json.load(tf.gfile.Open(bagging_odds_file))
    final_odds = {}
    for k in keys:
        final_odds[k] = (pred_odds[k] + 1 / (1 + math.exp(- bagging_odds[k]))) / 2
    json.dump(final_odds, tf.gfile.Open(output_odds_file, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    predict()
