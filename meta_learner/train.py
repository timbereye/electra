#!/usr/bin/env python
# encoding:utf-8
# -----------------------------------------#
# Filename:     train.py
#
# Description:
# Version:      1.0
# Created:      2020/7/16 15:22
# Author:       chenxiang@myhexin.com
# Company:      www.iwencai.com
#
# -----------------------------------------#

import os
import json
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import joblib
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf
import itertools
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier


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
        print("Samples Num: ".format(len(keys)))

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


class LR:
    def __init__(self, model_dir):
        if not tf.gfile.Exists(model_dir):
            tf.gfile.MakeDirs(model_dir)
        self.model_dir = model_dir

    def build_model(self, params):
        return LogisticRegression(**params, n_jobs=-1)

    def train_single(self, X, Y, params):
        clf = self.build_model(params)
        clf.fit(X, Y)
        joblib.dump(clf, os.path.join(self.model_dir, "lr.pkl"))

    def grid_search(self, train_X, train_Y, val_X, val_Y, keys_pred):
        grid = {
            "solver": ["liblinear", "lbfgs", "newton-cg", "sag"],
            "C": [0.01, 0.1, 1, 10, 100],
            "class_weight": [{0:1, 1:3}, {0:1, 1:2}, {0:1, 1:5}, {0:1, 1:6}],
            # "max_iter": [100, 500, 1000, 2000]
        }
        params_list = []
        for v1, v2, v3 in itertools.product(grid["solver"], grid["C"], grid["class_weight"]):
            params_list.append({"solver": v1, "C": v2, "class_weight": v3})
        best_params = None
        best_report = None
        best_model = None
        best_f1 = .0
        log = []
        for params in params_list:
            clf = self.build_model(params)
            clf.fit(train_X, train_Y)
            pred = clf.predict(val_X)
            report = classification_report(val_Y, pred, digits=5)
            print("params: {}  ---  val report:\n{}\n".format(params, report))
            f1 = f1_score(val_Y, pred)
            log.append((params, f1))
            if f1 > best_f1:
                best_f1 = f1
                best_report = report
                best_params = params
                best_model = clf
        print("Final result:\nbest params:\n{}\nbest report:\n{}\n".format(best_params, best_report))
        joblib.dump(best_model, os.path.join(self.model_dir, "lr.pkl"))
        log = sorted(log, key=lambda x: x[1], reverse=True)
        with open("lr_log_{}.txt".format(time.time()), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False)
        with open("lr_best_pred.txt", "w", encoding="utf-8") as f:
            json.dump(best_pred(best_model, val_X, keys_pred), f, ensure_ascii=False)


class RF:
    def __init__(self, model_dir):
        if not tf.gfile.Exists(model_dir):
            tf.gfile.MakeDirs(model_dir)
        self.model_dir = model_dir

    def build_model(self, params):
        return RandomForestClassifier(**params, n_jobs=-1)

    def train_single(self, X, Y, params):
        clf = self.build_model(params)
        clf.fit(X, Y)
        joblib.dump(clf, os.path.join(self.model_dir, "rf.pkl"))

    def grid_search(self, train_X, train_Y, val_X, val_Y, keys_pred):
        grid = {
            "n_estimators": [100, 500],
            "max_depth": [None, 3, 5],
            "min_samples_split": [10, 15, 20],
            "min_samples_leaf": [7, 10, 15],
            "class_weight": [{0: 1, 1: 3}, {0:1, 1:5}, {0:1, 1:7}],
        }
        params_list = []
        for v1, v2, v3, v4, v5 in itertools.product(grid["n_estimators"], grid["max_depth"], grid["min_samples_split"],
                                                    grid["min_samples_leaf"], grid["class_weight"]):
            params_list.append({"n_estimators": v1, "max_depth": v2, "min_samples_split": v3, "min_samples_leaf": v4,
                                "class_weight": v5})
        best_params = None
        best_report = None
        best_model = None
        best_f1 = .0
        log = []
        for params in params_list:
            clf = self.build_model(params)
            clf.fit(train_X, train_Y)
            pred = clf.predict(val_X)
            report = classification_report(val_Y, pred, digits=5)
            print("params: {}  ---  val report:\n{}\n".format(params, report))
            f1 = f1_score(val_Y, pred)
            log.append((params, f1))
            if f1 > best_f1:
                best_f1 = f1
                best_report = report
                best_params = params
                best_model = clf
        print("Final result:\nbest params:\n{}\nbest report:\n{}\n".format(best_params, best_report))
        joblib.dump(best_model, os.path.join(self.model_dir, "rf.pkl"))
        log = sorted(log, key=lambda x: x[1], reverse=True)
        with open("rf_log_{}.txt".format(time.time()), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False)
        with open("rf_best_pred.txt", "w", encoding="utf-8") as f:
            json.dump(best_pred(best_model, val_X, keys_pred), f, ensure_ascii=False)


class XGB:
    def __init__(self, model_dir):
        if not tf.gfile.Exists(model_dir):
            tf.gfile.MakeDirs(model_dir)
        self.model_dir = model_dir

    def build_model(self, params):
        return XGBClassifier(**params, n_jobs=-1)

    def train_single(self, X, Y, params):
        clf = self.build_model(params)
        clf.fit(X, Y)
        joblib.dump(clf, os.path.join(self.model_dir, "xgb.pkl"))

    def grid_search(self, train_X, train_Y, val_X, val_Y, keys_pred):
        grid = {
            "n_estimators": [100, 500, 1000],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.001, 0.01, 0.1],
            "scale_pos_weight": [3, 5, 7],
            "gamma": [0],
            # "max_delta_step": [5, 10, 20],
        }
        params_list = []
        for v1, v2, v3, v4, v5 in itertools.product(grid["n_estimators"], grid["max_depth"], grid["learning_rate"],
                                                        grid["scale_pos_weight"], grid["gamma"]):
            params_list.append({"n_estimators": v1, "max_depth": v2, "learning_rate": v3, "scale_pos_weight": v4,
                                "gamma": v5})
        best_params = None
        best_report = None
        best_model = None
        best_f1 = .0
        log = []
        for params in params_list:
            clf = self.build_model(params)
            clf.fit(train_X, train_Y)
            pred = clf.predict(val_X)
            report = classification_report(val_Y, pred, digits=5)
            print("params: {}  ---  val report:\n{}\n".format(params, report))
            f1 = f1_score(val_Y, pred)
            log.append((params, f1))
            if f1 > best_f1:
                best_f1 = f1
                best_report = report
                best_params = params
                best_model = clf
        print("Final result:\nbest params:\n{}\nbest report:\n{}\n".format(best_params, best_report))
        joblib.dump(best_model, os.path.join(self.model_dir, "xgb.pkl"))
        log = sorted(log, key=lambda x: x[1], reverse=True)
        with open("xgb_log_{}.txt".format(time.time()), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False)
        with open("xgb_best_pred.txt", "w", encoding="utf-8") as f:
            json.dump(best_pred(best_model, val_X, keys_pred), f, ensure_ascii=False)


def stacking(model_dir, train_X, train_Y, val_X, val_Y, keys_pred):
    estimators = [
        ("rl", LogisticRegression()),
        ("rf", RandomForestClassifier()),
        ("xgb", XGBClassifier())
    ]
    stk_func = lambda params: StackingClassifier(estimators, final_estimator=LogisticRegression(**params))
    grid = {
        "solver": ["liblinear", "lbfgs", "newton-cg", "sag"],
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [{0: 1, 1: 3}, {0: 1, 1: 2}, {0: 1, 1: 5}, {0: 1, 1: 6}],
    }
    params_list = []
    for v1, v2, v3 in itertools.product(grid["solver"], grid["C"], grid["class_weight"]):
        params_list.append({"solver": v1, "C": v2, "class_weight": v3})
    best_params = None
    best_report = None
    best_model = None
    best_f1 = .0
    for params in params_list:
        clf = stk_func(params)
        clf.fit(train_X, train_Y)
        pred = clf.predict(val_X)
        report = classification_report(val_Y, pred)
        print("val report:\n{}\n".format(report))
        f1 = f1_score(val_Y, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_report = report
            best_params = params
            best_model = clf
    print("Final result:\nbest params:\n{}\nbest report:\n{}\n".format(best_params, best_report))
    joblib.dump(best_model, os.path.join(model_dir, "stk.pkl"))


def best_pred(model, X, keys):
    pred = model.predict_proba(X)[:, 1]
    ret = {}
    for i, key in enumerate(keys):
        ret[key] = float(pred[i])
    return ret


def main(mode=0):
    model_dir = "./models/"
    pred_files_pattern_train = [
        "./data/bs32_seq384_lr5e-05_ep2.0_train.json",
        "./data/bs32_seq512_lr3e-05_ep3_train.json",
        "./data/bs32_seq512_lr5e-05_ep2.0_train.json",
        "./data/albert_pv_2_384_2e-5_train.json",
        # "./data/albert_pv_2_512_5e-5_train.json"
    ]
    pred_files_pattern_val = [
        "./data/bs32_seq384_lr5e-05_ep2.0_dev.json",
        "./data/bs32_seq512_lr3e-05_ep3_dev.json",
        "./data/bs32_seq512_lr5e-05_ep2.0_dev.json",
        "./data/albert_pv_2_384_2e-5_dev.json",
        # "./data/albert_pv_2_512_5e-5_dev.json"

    ]
    label_file_train = "./data/label_train.json"
    label_file_val = "./data/label_dev.json"
    train_data_loader = DataLoader(tf.gfile.Glob(pred_files_pattern_train), label_file_train)
    train_X, train_Y, _ = train_data_loader.load(do_shuffle=True)
    val_data_loader = DataLoader(tf.gfile.Glob(pred_files_pattern_val), label_file_val)
    val_X, val_Y, keys_pred = val_data_loader.load()
    if mode == 0:
        # LR
        print("************* LR *************")
        # lr = LR(model_dir)
        # lr.grid_search(train_X, train_Y, val_X, val_Y, keys_pred)
        # # RF
        # print("************* RF *************")
        # rf = RF(model_dir)
        # rf.grid_search(train_X, train_Y, val_X, val_Y, keys_pred)
        # XGB
        print("************* XGB *************")
        xgb = XGB(model_dir)
        xgb.grid_search(train_X, train_Y, val_X, val_Y, keys_pred)
    if mode == 1:
        stacking(model_dir, train_X, train_Y, val_X, val_Y, keys_pred)


if __name__ == '__main__':
    main()

"""
############## log #################


# RF




# XGB



"""