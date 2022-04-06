import logging
from test.rules.utils.load_dataset import load_dataset

from src.rules.intention.aux_slice.dispatch import dispatch
from src.utils.spacy import char_idx_to_word_idx, get_spacy

logger = logging.getLogger(__name__)


# proper = True, proper subset, same borders return False
def range_include(
    range_big: range, range_small: range, proper: bool = False
) -> bool:
    # [()]
    x1, x2 = range_big.start, range_big.stop
    y1, y2 = range_small.start, range_small.stop
    if proper:
        if x1 == y1 and x2 == y2:
            return False
    return x1 <= y1 <= y2 <= x2


def range_across(range1: range, range2: range) -> bool:
    x1, x2 = range1.start, range1.stop
    y1, y2 = range2.start, range2.stop
    # [(])
    pattern1 = x1 < y1 < x2 < y2
    # ([)]
    pattern2 = y1 < x1 < y2 < x2
    return pattern1 or pattern2


def range_disjoint(range1: range, range2: range) -> bool:
    x1, x2 = range1.start, range1.stop
    y1, y2 = range2.start, range2.stop
    pattern1 = x1 <= x2 <= y1 <= y2
    pattern2 = y1 <= y2 <= x1 <= x2
    return pattern1 or pattern2


def test_smoke():
    nlp = get_spacy()
    data = list(load_dataset("pretrained_data/task_core_aux_cond/all.jsonl"))
    logger.info("Start acl_without_to smoke test")
    sents = [d[1] for d in data]
    for sent in sents:
        logger.debug(sent)
        s = nlp(sent)[:]
        result = dispatch(s)
        logger.debug(result)


# TRUE
# NotAux: specify it is an Aux, but actually there is a Core
# e.g. Predict: Aux ..., Ground-truth: Core ...
# ShouldAux: specify it is a Core, but actually there is an Aux
# e.g. Predict: Core ..., Ground-truth: Aux ...
# SLICE_MISTAKE: the label anno across the slices
def test_how_slices_hit():
    core: str = "Core"
    aux: str = "Aux"
    nlp = get_spacy()
    data = list(load_dataset("pretrained_data/task_core_aux_cond/all.jsonl"))
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]
    total_true, total_not_aux, total_should_aux = 0, 0, 0
    for idx, (sent, label) in enumerate(zip(sents, labels)):
        logger.debug(sent)
        s = nlp(sent)[:]
        result = dispatch(s)
        # slice_mistake = 0
        # r_range [......][......]  # predict
        # l_range [..(.)...(...).]  # ground-truth
        for iidx, r in enumerate(result):
            true, not_aux, should_aux = 0, 0, 0
            r_start, r_end, r_anno = r
            r_range = range(r_start, r_end + 1)
            for lab in label:
                l_start, l_end, l_anno = lab
                l_start, l_end = char_idx_to_word_idx(s, l_start, l_end)
                l_range = range(l_start, l_end)
                # `cond as aux`
                # if l_anno == cond:
                #     l_anno = aux
                if l_anno not in [core, aux]:
                    continue

                if range_include(r_range, l_range):
                    if l_anno == r_anno == aux:
                        true += 1
                    elif l_anno == core and r_anno == aux:
                        not_aux += 1
                    elif l_anno == aux and r_anno == core:
                        should_aux += 1
                elif range_disjoint(r_range, l_range):
                    pass

                if not_aux or should_aux:
                    logger.debug(
                        f"predict slice range: {(r_start, r_end + 1)} - {s[r_start : r_end + 1]} - {r_anno}"
                    )
                    logger.debug(
                        f"ground label range: {(l_start, l_end)} - {s[l_start : l_end]} - {l_anno}"
                    )
                    for to in s:
                        if to.dep_ == "nsubj":
                            logger.debug(f"Root: {to.head}")

            if true:
                logger.info(f"#{idx}-{iidx} True: {true}")
            if not_aux:
                logger.warning(f"#{idx} Not aux: {not_aux}")
            if should_aux:
                logger.warning(f"#{idx} Should aux: {should_aux}")

            total_true += true
            total_not_aux += not_aux
            total_should_aux += should_aux

    logger.error(f"{total_true}/{total_not_aux}/{total_should_aux}")


# aux_without_to 194/63/122
# awt is 198/72/118
# awt relcl double != to 205/78/111
# awt relcl double != to is 194/63/122
# awt rd!=to is advcl 179/58/137
# awt rd!=to is advcl[0] 194/62/122
# awt rd!=to advcl[0] 205/73/111
# awt rd!=to advcl[0] cond as aux 215/73/202   # cond rules demand
# awt rd!=to auxpass advcl[0] 199/129/102
# awt rd!=to auxpass advcl[0] caa 220/130/182
# awt rd!=to nsubjpass advcl[0] 201/124/102
# awt!=using rd!=to advcl[0] 204/62/112
# awt!=using rd!=to is advcl[0] 193/51/123
# awt!=using rd!=to is[-1] advcl[0] 193/53/123
# awt!=using rd!=to is[0] advcl[0] 204/60/112
# awt!=using rd!=to agent is[0] advcl[0] 212/72/112
# awt!=using rd!=to agent is[-1] advcl[0] 203/67/121
# awt!=using rd!=to agent advcl[0] 213/76/111
# awt!=using rd!=to agent is[0] advcl[0] fix_pos 216/72/108
