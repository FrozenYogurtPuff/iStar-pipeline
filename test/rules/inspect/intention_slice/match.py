import logging
from test.rules.inspect.utils import cache_nlp, dep_list, if_inside
from test.rules.utils.load_dataset import load_dataset

from src.utils.spacy import get_spacy
from src.utils.typing import FixIntentionLabel

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    data = list(load_dataset("pretrained_data/task_core_aux_cond/all.jsonl"))

    nlp = get_spacy()
    aux: FixIntentionLabel = "Aux"

    for check in dep_list:
        total, match = 0, 0
        for i, sent, anno in data:
            sent_processed = cache_nlp(nlp, sent)
            if if_inside(sent_processed, check):
                total += 1
                for start, end, lab in anno:
                    if lab == aux:
                        match += 1
                        break

        if total > 0 and match / total > 0.4:
            logger.error(
                f"{check} - Match {match} out of {total} with {match / total if total else 0}"
            )
        else:
            logger.warning(
                f"{check} - Match {match} out of {total} with {match / total if total else 0}"
            )

# agent - Match 41 out of 55 with 0.7454545454545455
# oprd - Match 5 out of 9 with 0.5555555555555556
# relcl - Match 130 out of 187 with 0.6951871657754011
