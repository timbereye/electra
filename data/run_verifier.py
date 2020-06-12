import json
from copy import deepcopy
import os
import pickle
import numpy as np
from functional import seq
import argparse


def run_verifier(input_file, data_dir, output_file):
    """
    run pv verifier and reg verifier
    Args:
        input_file:
        data_dir:
        output_file: final prediction file

    Returns:

    """

    preds_file = os.path.join(data_dir, 'squad_preds.json')
    null_odds_file = os.path.join(data_dir, 'squad_null_odds.json')
    eval_file = os.path.join(data_dir, 'squad_eval.json')
    pv_null_odds_file = os.path.join(data_dir, 'pv_squad_null_odds.json')
    answer_candidates_score_file = os.path.join(data_dir, 'dev_f1_predict_results.pkl')
    dev_all_nbest_file = os.path.join(data_dir, 'dev_all_nbest.pkl')
    tmp_eval_file = os.path.join(data_dir, 'tmp_eval.json')
    tmp_preds_file = os.path.join(data_dir, 'tmp_preds.json')
    tmp_null_odds_file = os.path.join(data_dir, 'tmp_null_odds.json')

    preds = json.load(open(preds_file, 'r', encoding='utf-8'))
    null_odds = json.load(open(null_odds_file, 'r', encoding='utf-8'))
    best_th = json.load(open(eval_file, 'r', encoding='utf-8'))["best_exact_thresh"]
    pv_null_odds = json.load(open(pv_null_odds_file, 'r', encoding='utf-8'))
    answer_candidates_score = pickle.load(open(answer_candidates_score_file, 'rb'))
    dev_all_nbest = pickle.load(open(dev_all_nbest_file, 'rb'))

    merge_null_odds = deepcopy(null_odds)

    for k, v in merge_null_odds.items():
        merge_null_odds[k] = null_odds[k] + pv_null_odds[k]
    json.dump(merge_null_odds, open(tmp_null_odds_file, 'w', encoding='utf-8'))

    xargs = f"python eval.py {input_file} {preds_file} --na-prob-file {tmp_null_odds_file} --out-file {tmp_eval_file}"
    os.system(xargs)

    new_sh = json.load(open(tmp_eval_file, 'r', encoding='utf-8'))["best_exact_thresh"]

    for k, v in merge_null_odds.items():
        if v > new_sh:
            preds[k] = ""

    length = 5
    believe_f1_th = 1.
    believe_prob_th = 0.3
    dev = json.load(open(input_file))
    for article in dev['data']:
        for paragraph in article["paragraphs"]:
            for qa in paragraph['qas']:
                qid = qa['id']
                answers = qa['answers']
                # if not answers:
                #     print(1)
                chooses = (seq(range(length))
                           .map(lambda x: answer_candidates_score.get(f"{qid}_{x}", None))
                           .map(lambda x: x if x is None else {'f1_pred': np.max([y['predictions'] for y in x])})
                           .zip(dev_all_nbest[qid][:length])
                           .filter(lambda x: x[0])
                           .map(lambda x: {'text': x[1]['text'],
                                           'f1_pred': x[0]['f1_pred'],
                                           'prob': x[1]['probability']})
                           .sorted(lambda x: x['f1_pred'], reverse=True)
                           ).list()
                if len(chooses):
                    max_prob = seq(chooses).map(lambda x: x['prob']).max()
                    if chooses[0]['f1_pred'] > believe_f1_th and max_prob - chooses[0]['prob'] < believe_prob_th and not preds[qid]:
                        # if not answers:
                        #     print(2)
                        # if not preds[qid] and answers:
                        #     print(3)
                        # if answers and preds[qid]:
                        #     print(4)
                        preds[qid] = chooses[0]['text']
                        print(f"{chooses[0]},    {len(answers) > 0}")

    # for qid in preds:
    #     chooses = (seq(range(length))
    #                .map(lambda x: answer_candidates_score.get(f"{qid}_{x}", None))
    #                .map(lambda x: x if x is None else {'f1_pred': np.max([y['predictions'] for y in x])})
    #                .zip(dev_all_nbest[qid][:length])
    #                .filter(lambda x: x[0])
    #                .map(lambda x: {'text': x[1]['text'],
    #                                'f1_pred': x[0]['f1_pred'],
    #                                'prob': x[1]['probability']})
    #                .sorted(lambda x: x['f1_pred'], reverse=True)
    #                ).list()
    #     if len(chooses):
    #         max_prob = seq(chooses).map(lambda x: x['prob']).max()
    #         if chooses[0]['f1_pred'] > believe_f1_th and max_prob - chooses[0]['prob'] < believe_prob_th and not preds[qid]:
    #             preds[qid] = chooses[0]['text']
    #             print(chooses[0])

    json.dump(preds, open(output_file, 'w', encoding='utf-8'))

    xargs = f"python eval.py {input_file} {output_file}"
    os.system(xargs)


def main():
    # parser = argparse.ArgumentParser()
    #
    # # Required parameters
    # parser.add_argument('--eval-file', required=True, help="eval file")
    # parser.add_argument('--data-dir', required=True, help="data dir")
    # parser.add_argument('--output-file', required=True, help="final predictions")
    # args = parser.parse_args()
    #
    # run_verifier(args.eval_file, args.data_dir, args.output_file)
    run_verifier("dev-v2.0.json", "./", "pp.json")


if __name__ == '__main__':
    main()
