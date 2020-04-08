import json
import os
import shutil


def ev(eval_file, preds_file, null_odds_file, eval_result_file, answer_null_odds_file, output_file):
    prediction = json.load(open(preds_file, 'r', encoding='utf-8'))
    null_odds = json.load(open(null_odds_file, 'r', encoding='utf-8'))
    best_th = json.load(open(eval_result_file, 'r', encoding='utf-8'))["best_exact_thresh"]
    answer_null_odds = json.load(open(answer_null_odds_file, 'r', encoding='utf-8'))

    tmp_file = 'tmp_preds.json'
    tmp_eval_file = 'tmp_eval.json'
    xargs = f"python ./finetune/qa/squad_official_eval.py {eval_file} {preds_file} " \
            f"--na-prob-file {null_odds_file} --na-prob-thresh {best_th}"
    os.system(xargs)
    for k, v in null_odds.items():
        if v > best_th:
            prediction[k] = ""
    json.dump(prediction, open(tmp_file, 'w', encoding='utf-8'))
    xargs = f"python ./finetune/qa/squad_official_eval.py {eval_file} {tmp_file} " \
            f"--na-prob-file {answer_null_odds_file} --out-file {tmp_eval_file}"
    os.system(xargs)
    new_sh = json.load(open(tmp_eval_file, 'r', encoding='utf-8'))["best_exact_thresh"]
    for k, v in answer_null_odds.items():
        if v > new_sh:
            prediction[k] = ""
    json.dump(prediction, open(output_file, 'w', encoding='utf-8'))
    xargs = f"python ./finetune/qa/squad_official_eval.py {eval_file} {output_file} "

    os.system(xargs)
    shutil.rmtree(tmp_file)
    shutil.rmtree(tmp_eval_file)
