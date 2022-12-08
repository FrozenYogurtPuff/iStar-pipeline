import spacy
from flask import Flask, jsonify, request
from flask_cors import CORS
from spacy_alignments import tokenizations

from src.deeplearning.entity.infer.utils import get_series_bio
from src.deeplearning.entity.infer.wrapper import ActorWrapper, \
    IntentionWrapper
from src.rules.intention.find_object import find_object
from src.deeplearning.relation.main_task import proceed_sentence

app = Flask(__name__)
CORS(app)

a_wrapper = ActorWrapper()
i_wrapper = IntentionWrapper()
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
            cur = list()
            spacy_start = b2s[start][0]
            spacy_end = b2s[end][-1]
            span = sent[spacy_start:spacy_end+1]
            cur.append((tag, span.start_char, span.end_char))
            for token in find_object(span):
                obj_i = token.i
                obj_span = sent[obj_i:obj_i+1]
                cur.append(("Obj", obj_span.start_char, obj_span.end_char))
            result.append(cur)
        print(result)
        return jsonify(result)


@app.route('/ar', methods=['POST'])
def actor_relation():
    if request.method == "POST":
        json = request.get_json()
        text = json.get("text")
        entities = json.get("entity")
        result = proceed_sentence(text, entities)
        print(result)
        return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050)
