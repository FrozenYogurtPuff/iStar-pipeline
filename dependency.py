import spacy
import en_core_web_lg

nlp = en_core_web_lg.load()


def getNounSpan(token):
    token_list = list()
    end = token.i - token.sent.start + 1
    border = token.sent.start
    for child in token.children:
        if child.dep_ in ['conj', 'appos']:
            token_list += getNounSpan(child)
    flag = True
    while token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ'] or token.dep_ == 'compound' or (
            token.tag_ == 'JJ' and token.dep_ in ['amod', 'advmod']):
        if token.i != border:
            token = token.nbor(-1)
        else:
            flag = False
            break
    if flag:
        token = token.nbor(1)
    start = token.i - token.sent.start
    token_list.append((start, end))
    return token_list


def simple_noun_chunks(doc):
    sentences = nlp(doc)
    preds_list = list()
    for sent in sentences.sents:
        preds_cur = list()
        chunks = list(sent.noun_chunks)
        for chunk in chunks:
            preds_cur.append((chunk.start - sent.start, chunk.end - sent.start - 1))
        preds_list.append(preds_cur)
    return preds_list


def pred_entity(doc):
    sentences = nlp(doc)
    results = list()
    tokens = list()
    preds = list()
    for sent in sentences.sents:
        tokens.append([i.text for i in sent])

        # Entity
        entity = []
        for idx, B in enumerate(sent):
            dep_word = B.dep_
            A = B.head
            # TDR 1 & 2
            if dep_word[:5] == 'nsubj':
                if A.pos_ == 'VERB' and B.pos_ in ['NOUN', 'PROPN']:  # TDR 1: B != Basic_Attrib
                    token_list = getNounSpan(B)
                    # print('#1-2', token_list)
                    entity += token_list
            if dep_word in ['dobj', 'iobj', 'pobj']:
                if A.pos_ == 'VERB' and B.pos_ in ['NOUN',
                                                   'PROPN']:  # TDR 3: B != Basic_Attrib and prevTD != ”amod” and prevTD != ”advmod” and VB != ”entered” | “inputted” | “saved” |”added” | “has”
                    token_list = getNounSpan(B)
                    # print('#3-5', token_list)
                    entity += token_list
            # TDR 6 & 7
            if B.lower_ in ['of', 'in']:
                pre = B.head
                post = list(B.children)[0]
                if pre.pos_ in ['NOUN', 'PROPN'] and post.pos_ in ['NOUN', 'PROPN']:
                    pre_list = getNounSpan(pre)
                    post_list = getNounSpan(post)
                    # print('#6-7', pre_list)
                    # print('#6-7', post_list)
                    entity += pre_list
                    entity += post_list
            # TDR 8
            if B.lower_ in ['to', 'for', 'from', 'as']:
                if len(list(B.children)):
                    post = list(B.children)[0]
                    if post.pos_ in ['NOUN', 'PROPN']:
                        post_list = getNounSpan(post)
                        # print('#8', post_list)
                        entity += post_list

            # TDR 9
            if B.lower_ in ['by', 'agent', 'with']:
                if len(list(B.children)):
                    post = list(B.children)[0]
                    if post.pos_ in ['NOUN', 'PROPN']:
                        post_list = getNounSpan(post)
                        # print('#9', post_list)
                        entity += post_list
            # TDR 10
            if dep_word == 'poss':
                if A.pos_ in ['NOUN', 'PROPN']:
                    A_list = getNounSpan(A)
                    # print('#10', A_list)
                    entity += A_list
                if B.pos_ in ['NOUN', 'PROPN']:
                    B_list = getNounSpan(B)
                    # print('#10', B_list)
                    entity += B_list

        entity = sorted(entity, key=lambda x: x[0])

        entity = [(en[0], en[1] - 1) for en in entity]

        # Remove duplicate
        cur_idx = 0
        while cur_idx < len(entity) - 1:
            cur_entity = entity[cur_idx]
            next_entity = entity[cur_idx + 1]
            if next_entity[1] <= cur_entity[1]:
                entity.pop(cur_idx + 1)
            cur_idx += 1

        result = ['O' for _ in range(len(sent))]
        for en in entity:
            for i, idx in enumerate(range(en[0], en[1] + 1)):
                if i == 0:
                    result[idx] = 'B'
                else:
                    result[idx] = 'I'

        results.append(result)
        preds.append(entity)
    return results, tokens, preds
