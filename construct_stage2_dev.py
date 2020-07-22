#!/usr/bin/env python
# encoding:utf-8
# -----------------------------------------#
# Filename:     construct_stage2_dev.py.py
#
# Description:
# Version:      1.0
# Created:      2020/7/22 15:34
# Author:       chenxiang@myhexin.com
# Company:      www.iwencai.com
#
# -----------------------------------------#

import fire
import json
import tensorflow as tf


def main(bagging_pred_file, dev_file, output_file):
    with tf.gfile.Open(bagging_pred_file) as f:
        bagging_pred = json.load(f)
    with tf.gfile.Open(dev_file) as f:
        data = json.load(f)
        for d in data["data"]:
            for para in d["paragraphs"]:
                for qas in para["qas"]:
                    qas["answers"] = []
                    qas["plausible_answers"] = [{"text": bagging_pred[qas["id"]], "answer_start": 0}]
    json.dump(data, tf.gfile.Open(output_file, 'w'), ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
