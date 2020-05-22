import json

dev = json.load(open('dev-v2.0.json'))
preds = json.load(open('atrlp8876_squad_preds.json'))

for article in dev['data']:
    for paragraph in article["paragraphs"]:
        context = paragraph['context']
        for qa in paragraph['qas']:
            qid = qa['id']
            pred = preds[qid]
            qa['is_impossible'] = True
            qa['plausible_answers'] = [{'text': pred, 'answer_start': 1}]

json.dump(dev, open('dev.json', 'w', encoding='utf-8'))
print("write finished !")
