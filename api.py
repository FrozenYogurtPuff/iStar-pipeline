import spacy
from flask import Flask, jsonify, request
from flask_cors import CORS
from spacy_alignments import tokenizations

from src.deeplearning.entity.infer.utils import get_series_bio
from src.deeplearning.entity.infer.wrapper import ActorWrapper, \
    IntentionWrapper
from src.rules.intention.find_object import find_object
from src.deeplearning.relation.main_task import proceed_sentence, preload

app = Flask(__name__)
CORS(app)

a_wrapper = ActorWrapper()
i_wrapper = IntentionWrapper()
infer = preload()
nlp = spacy.load("en_core_web_lg")


@app.route('/ae', methods=['POST'])
def actor_entity():
    if request.method == "POST":
        json = request.get_json()
        text = json.get("text")

        results = a_wrapper.process(text)
        pred_entities, _ = get_series_bio(results)
        print(pred_entities)
        result = list()
        b_tokens = results[0].tokens
        sent = nlp(text)
        s_tokens = [i.text for i in sent]
        _, b2s = tokenizations.get_alignments(s_tokens, b_tokens)
        for tag, start, end in pred_entities:
            spacy_start = b2s[start][0]
            spacy_end = b2s[end][-1]
            span = sent[spacy_start:spacy_end+1]
            result.append((tag, span.start_char, span.end_char))
        print(result)
        return jsonify(result)


@app.route('/ie', methods=['POST'])
def intention_entity():
    if request.method == "POST":
        json = request.get_json()
        text = json.get("text")

        results = i_wrapper.process(text)
        pred_entities, _ = get_series_bio(results)
        print(pred_entities)
        result = list()
        b_tokens = results[0].tokens
        sent = nlp(text)
        s_tokens = [i.text for i in sent]
        _, b2s = tokenizations.get_alignments(s_tokens, b_tokens)
        for tag, start, end in pred_entities:
            if tag != "Core":
                continue
            spacy_start = b2s[start][0]
            spacy_end = b2s[end][-1]
            span = sent[spacy_start:spacy_end+1]
            # TODO: 推入普通字符串而非 Verb - Obj
            # TODO: 判断 Goal (1) 与 Quality (2)
            # cur.append((tag, span.start_char, span.end_char))
            start_char = span.start_char
            end_char = span.end_char
            for token in find_object(span):
                obj_i = token.i
                obj_span = sent[obj_i:obj_i+1]
                if obj_span.start_char < start_char:
                    start_char = obj_span.start_char
                if obj_span.end_char > end_char:
                    end_char = obj_span.end_char
                # cur.append(("Obj", obj_span.start_char, obj_span.end_char))

            need_add = True
            for res in result:
                # 如果新的覆盖旧的，则弹旧加新
                if res[1] >= start_char and res[2] <= end_char:
                    result.remove(res)
                    break

                # 如果旧的覆盖新的，则不添加
                elif res[1] <= start_char and res[2] >= end_char:
                    need_add = False
                    break

            if need_add:
                result.append(("Goal", start_char, end_char))
                print(sent.char_span(start_char, end_char))

        print(result)
        return jsonify(result)


@app.route('/ar', methods=['POST'])
def actor_relation():
    if request.method == "POST":
        json = request.get_json()
        text = json.get("text")
        entities = json.get("entity")
        result = proceed_sentence(text, entities, infer)
        print(result)
        return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050)
