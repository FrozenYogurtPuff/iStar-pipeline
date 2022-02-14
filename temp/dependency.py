import spacy
import en_core_web_lg

nlp = en_core_web_lg.load()


def notStart(token):
    return token.i - token.sent.start != 0


def notEnd(token):
    return token.sent.end - token.i > 1


def getFullSpan(token):
    sent = token.sent
    if token.pos_ in ['NOUN', 'PROPN', 'PRON']:  # Pronoun for 'I'
        token_list = getNounSpan(token)
    # elif token.pos_ in ['VERB', 'ADV', 'AUX']:
    else:
        token_list = getVerbSpan(token)
    # else:
    #     raise Exception("Illegal " + token.lower_ + " with " + token.pos_ + " " + token.tag_ + " " + token.dep_)
    return [sent[token[0]:token[1]] for token in sorted(token_list, key=lambda x: x[1])]


def getVerbSpan(token):
    token_list = list()
    end = token.i - token.sent.start + 1
    for child in token.children:
        if child.dep_ == 'conj':
            token_list += getVerbSpan(child)
    start = token.i - token.sent.start
    token_list.append((start, end))
    return token_list


def getNounSpan(token):
    token_list = list()
    end = token.i - token.sent.start + 1
    border = token.sent.start
    for child in token.children:
        if child.dep_ in ['conj', 'appos']:
            token_list += getNounSpan(child)
    flag = False
    while token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ'] or token.dep_ == 'compound' or (
            token.tag_ == 'JJ' and token.dep_ in ['amod', 'advmod']):
        if token.i != border:
            flag = True
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
                                                   'PROPN']:
                    # TDR 3: B != Basic_Attrib and prevTD != ”amod” and prevTD != ”advmod” and VB != ”entered” | “inputted” | “saved” |”added” | “has”
                    token_list = getNounSpan(B)
                    # print('#3-5', token_list)
                    entity += token_list
            # TDR 6 & 7
            if B.lower_ in ['of', 'in']:
                pre = B.head
                if len(list(B.children)) != 0:
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


def findChildren(token, dep=None, pos=None, tag=None, text=None):
    children_list = list()
    if len(list(token.children)) == 0:
        return None
    for t in token.children:
        flag = True
        if dep and t.dep_ != dep:
            flag = False
        if pos and t.pos != pos:
            flag = False
        if tag and t.tag != tag:
            flag = False
        if text and t.lower_ != text:
            flag = False
        if flag:
            children_list.append(t)
    return children_list


def makeRelationDict():
    dic = {'subject': list(), 'verb': list(), 'object': list(), 'prep': list(), 'adv': list() }
    return dic


# TODO: 串联仍然存在问题
def searchVerb(token_list, main=False, already=None):
    relation_list = list()
    # imperative = True
    if main:
        verb_list = list()
        # acl
        for token in token_list:
            if token.dep_ in ['acl', 'pcomp', 'relcl']:
                verb_list.append(token)
        for token in token_list:
            if token.dep_ in ['nsubj', 'nsubjpass']:
                verb_list.extend([t[0] for t in getFullSpan(token.head)])
                # imperative = False
                break
        token_list = verb_list
    if not already:
        already = list()
    else:
        if token_list in already:
            return list()
    already.append(token_list)
    for verb in token_list:
        if verb.pos_ not in ['VERB', 'AUX']:
            continue
        r_dict = makeRelationDict()
        # nsubj
        if findChildren(verb, dep='nsubj'):
            subj_list = list()
            subj_temp_list = findChildren(verb, dep='nsubj')
            for subj in subj_temp_list:
                subj_list.append(getFullSpan(subj))
            r_dict['subject'] = subj_list
        # nsubjpass
        if findChildren(verb, dep='nsubjpass'):
            obj_list = list()
            obj_temp_list = findChildren(verb, dep='nsubjpass')
            for obj in obj_temp_list:
                obj_list.append(getFullSpan(obj))
            r_dict['object'] = obj_list
            # by
            if findChildren(verb, dep='agent'):
                agent = findChildren(verb, dep='agent')[0]
                subj_list = getFullSpan(list(agent.children)[0])
                r_dict['subject'] = subj_list
        # adv
        if findChildren(verb, dep='advmod'):
            adv_list = findChildren(verb, dep='advmod')
            for adv in adv_list:
                r_dict['adv'].append(adv)
        # verb prep phrase
        if findChildren(verb, dep='prep'):
            flag = False
            for prep in findChildren(verb, dep='prep'):
                if notStart(prep) and prep.nbor(-1) == verb:
                    flag = True
                    verb_prep = prep
                    break
            if flag:
                r_dict['verb'].append({'v': verb, 'prep': verb_prep})
                verb = verb_prep
            else:
                r_dict['verb'].append({'v': verb, 'prep': None})
        else:
            r_dict['verb'].append({'v': verb, 'prep': None})
        # dobj, pobj
        if findChildren(verb, dep='dobj') or findChildren(verb, dep='pobj'):
            obj_list = findChildren(verb, dep='dobj') + findChildren(verb, dep='pobj')
            for child in obj_list:
                child_span = getFullSpan(child)
                r_dict['object'].append(child_span)
                child = child_span[-1][-1]
                # prep
                while notEnd(child) and child.nbor(1).dep_ == 'prep' and findChildren(child.nbor(1), dep='pobj'):
                    post = findChildren(child.nbor(1), dep='pobj')
                    for p in post:
                        r_dict['prep'].append({'p': child.nbor(1), 'obj': getFullSpan(p)})
                    child = post[-1]
        # comp, advcl
        relation_list.append(r_dict)
        if findChildren(verb, 'xcomp') or findChildren(verb, 'ccomp') or findChildren(verb, 'pcomp') or findChildren(
                verb, 'advcl'):
            comp_list = findChildren(verb, 'xcomp') + findChildren(verb, 'ccomp') + findChildren(verb,
                                                                                                 'pcomp') + findChildren(
                verb, 'advcl')
            result = searchVerb(comp_list, already=already)
            for r in result:
                relation_list.append(r)
        if verb.dep_ in ['xcomp', 'ccomp', 'pcomp', 'advcl']:
            verb_head = [t[0] for t in getFullSpan(verb.head)]
            result = searchVerb(verb_head, already=already)
            for r in result:
                relation_list.append(r)
    return relation_list


# TODO: Dependee 查找与 Entity 对应
def extractTask(r_dict):
    r_dict['subject'] = list()
    return r_dict


def getPosition(r_dict, minn=1e10, maxx=-1):
    if isinstance(r_dict, dict):
        for key in r_dict:
            minn, maxx = getPosition(r_dict[key], minn, maxx)
    elif isinstance(r_dict, tuple):
        minn, maxx = getPosition(r_dict[0], minn, maxx)
    elif r_dict == None:
        return minn, maxx
    elif isinstance(r_dict, list):
        for item in r_dict:
            minn, maxx = getPosition(item, minn, maxx)
    elif isinstance(r_dict, spacy.tokens.token.Token):
        begin = r_dict.i - r_dict.sent.start
        end = r_dict.i - r_dict.sent.start + 1
        if end > maxx:
            maxx = end
        if begin < minn:
            minn = begin
    elif isinstance(r_dict, spacy.tokens.span.Span):
        begin = r_dict.start
        end = r_dict.end
        if end > maxx:
            maxx = end
        if begin < minn:
            minn = begin
    return (minn, maxx)

# TODO: 结构化而非span式的支持
def pred_task(doc):
    sentences = nlp(doc)
    ret = list()
    for s in sentences.sents:
        ret_cur = list()
        res = searchVerb(s, True)
        for r in res:
            c = extractTask(r)
            a, b = getPosition(c)
            ret_cur.append((a, b))
        ret_cur = sorted(ret_cur, key=lambda x: x[0])
        ret_cur = [(en[0], en[1] - 1) for en in ret_cur]
        # Remove duplicate
        cur_idx = 0
        while cur_idx < len(ret_cur) - 1:
            cur_entity = ret_cur[cur_idx]
            next_entity = ret_cur[cur_idx + 1]
            if next_entity[1] <= cur_entity[1]:
                ret_cur.pop(cur_idx + 1)
            cur_idx += 1
        ret.append(ret_cur)
    return ret
