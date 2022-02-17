import logging

from src.deeplearning.infer.entity import entity_model
from src.deeplearning.infer.intention import intention_model

logger = logging.getLogger(__name__)


def wrap_oneline(sent, labels=None):
    data = {'sent': sent}
    if labels:
        data['labels'] = labels
    return data


def unwrap_oneline(data):
    return [item[0] for item in data]


def wrap_entity_oneline(sent, labels=None):
    logger.debug(f'Entity one-line input: {sent}')
    data = wrap_oneline(sent, labels)
    result = entity_model.predict([data])
    logger.debug(f'Entity one-line output result: {result[0]}\n'
                 f'Entity one-line tokens: {result[3]}')
    return unwrap_oneline(result)


def wrap_intention_oneline(sent, labels=None):
    data = wrap_oneline(sent, labels)
    result = intention_model.predict([data])
    return unwrap_oneline(result)
