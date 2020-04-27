import json
from functional import seq

ambig_train = json.load(open('train_light.json', 'r', encoding='utf-8'))

nq_train_f = open('/content/simplified-nq-train.jsonl', 'r', encoding='utf-8')

id2ambig = seq(ambig_train).map(lambda x: [int(x['id']), x]).dict()

count = 0
for line in nq_train_f:
    nq = json.loads(line)
    nq_id = nq['example_id']
    if nq_id in id2ambig:
        id2ambig[nq_id]['context'] = nq['document_text']
    count += 1
    if count % 1000 == 0:
        print(f"finished processing {count}")

json.dump(list(id2ambig.values()), open('ambig_train_with_context.json', 'w', encoding='utf-8'))
