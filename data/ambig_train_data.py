import json
from finetune.qa.squad_official_eval import normalize_answer
from bs4 import BeautifulSoup
from functional import seq
from fuzzywuzzy import fuzz
import string

ambig_train_with_context = json.load(open('ambig_train_with_context.json', 'r', encoding='utf-8'))


def fix_content(s: str):
    s = s.replace(' ,', ',').replace(' .', '.').replace(' \'\'', '"').replace('`` ', '"').replace(' !', '!').replace(
        ' \'s', '\'s').replace(' - ', '-').replace(' -- ', '--').replace('( ', '(').replace(' )', ')')

    return s


for ambig in ambig_train_with_context:
    qa_type = ambig['annotations'][0]['type']
    if qa_type == 'multipleQAs':
        qa_pairs = ambig['annotations'][0]['qaPairs']
        context_list = seq(BeautifulSoup(ambig['context']).stripped_strings).map(fix_content).list()
        context_ids_weight = [0] * len(context_list)

        answer_appear_map = {}
        for i, qa in enumerate(qa_pairs):
            question = qa['question']
            answer = qa['answer'][0]
            answer_appear_map[i] = (seq(context_list)
                                    .enumerate().filter(lambda x: answer in x[1])
                                    .map(lambda x: (x[0], fuzz.token_set_ratio(question, x[1])))).list()
            for _id, _v in answer_appear_map[i]:
                context_ids_weight[_id] += _v
        score2span = {}
        score = 0
        span = []
        for _id, _v in enumerate(context_ids_weight):
            if _v == 0:
                if span:
                    span.append(_id)
                    score2span[score] = [min(span), max(span)]
                    span = []
                    score = 0
                continue
            score += _v
            span.append(_id)

        if score2span:
            final_span = (seq(score2span.items())
                          .sorted(lambda x: x[0], reverse=True)
                          ).list()[0][1]

            context = "\n".join(context_list[final_span[0]:final_span[1]])
            ambig['context'] = context
        else:
            ambig['context'] = None
print(1)
