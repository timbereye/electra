import json
import pickle
from finetune.qa.squad_official_eval import compute_f1
import argparse


def gen_pv_data(std_dev_file, preds_file, output_file):
    """
    generate data for plausible answer verifier
    Args:
        std_dev_file: official dev file
        preds_file: atrlp model prediction file
        output_file:

    Returns: a file

    """
    dev = json.load(open(std_dev_file, 'r', encoding='utf-8'))
    preds = json.load(open(preds_file, 'r', encoding='utf-8'))

    for article in dev['data']:
        for paragraph in article["paragraphs"]:
            for qa in paragraph['qas']:
                qid = qa['id']
                pred = preds[qid]
                qa['is_impossible'] = True
                qa['plausible_answers'] = [{'text': pred, 'answer_start': 1}]

    json.dump(dev, open(output_file, 'w', encoding='utf-8'))
    print("generate pv data finished! ")


def gen_answer_refine_file(std_dev_file, nbest_file, output_file):
    """
    generate answer refine file, for choose refine answer
    Args:
        std_dev_file: official dev file
        nbest_file: atrlp prediction nbest file

    Returns: a file

    """
    data = json.load(open(std_dev_file, 'r', encoding='utf-8'))
    all_nbest = pickle.load(open(nbest_file, 'rb'))
    split = 'dev'
    count = 0

    for article in data['data']:
        for p in article['paragraphs']:
            # del p['context']
            new_qas = []
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = qa['answers']
                if not gold_answers:
                    continue
                nbest = all_nbest[qid][:5]

                most_text = nbest[0]['text']
                new_qa = []
                for i, nb in enumerate(nbest):
                    pred = nb['text']
                    if split == 'train':
                        a = qa['answers'][0]['text']
                        f1 = compute_f1(a, pred)
                    else:
                        f1 = max(compute_f1(a['text'], pred) for a in gold_answers)
                    if pred in most_text or most_text in pred:
                        new_qa.append({"f1_score": f1,
                                       "pred_answer": pred,
                                       "question": qa['question'],
                                       "id": f"{qid}_{i}"})
                if new_qa[0]["f1_score"] > 0:
                    new_qas.extend(new_qa)
            p['qas'] = new_qas
            count += len(new_qas)

    print(count)

    json.dump(data, open(output_file, 'w', encoding='utf-8'))
    print("generate answer refine file finished! ")


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--run-type', required=True, help="Generate data type : pv or reg")
    parser.add_argument('--std-dev-file', required=True, help="Official eval file")
    parser.add_argument('--input-file', required=True, help="Previous model output ")
    parser.add_argument("--output-file", required=True, help="Generate data output")
    args = parser.parse_args()

    if args.run_type == 'pv':
        gen_pv_data(args.std_dev_file, args.input_file, args.output_file)
    elif args.run_type == 'reg':
        gen_answer_refine_file(args.std_dev_file, args.input_file, args.output_file)
    else:
        raise


if __name__ == '__main__':
    main()
