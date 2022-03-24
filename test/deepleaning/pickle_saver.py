import pickle
from test.rules.utils.load_dataset import load_dataset

from src.deeplearning.infer.wrapper import ActorWrapper


def actor_split_dev_bert_save():
    data = list(
        load_dataset("pretrained_data/2022/actor/divided/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]
    wrapper = ActorWrapper()
    results = wrapper.process(sents, labels)
    with open("actor_split_dev.bin", "wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    actor_split_dev_bert_save()
