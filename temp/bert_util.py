def calc_specific_prob(matrix, labels, label, start, end):
    label_reverse = {label: i for i, label in enumerate(labels)}
    b_idx = label_reverse['B-' + label]
    i_idx = label_reverse['I-' + label]
    prob = max(matrix[start][b_idx], matrix[start][i_idx])
    for i in range(start + 1, end + 1):
        prob += matrix[i][i_idx]
    prob /= (end - start + 1)
    return prob


def get_entities_bio_with_prob(seq, matrix, labels):
    """Gets entities from sequence with probability.
    note: BIO
    Args:
        seq (list): sequence of labels.
        matrix (list)
        labels (list)
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
        list: list of (prob)
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entities_bio_with_prob(seq, matrix, labels)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
        [0.92, 0.95]
    """

    def _add_to_chunks(chunk):
        prob = calc_specific_prob(matrix, labels, *chunk)
        chunks.append(chunk)
        probs.append(prob)

    if any(isinstance(s, list) for s in seq):
        raise Exception('seq is nested')
        # seq = [item for sublist in seq for item in sublist + ['O']]

    chunks = []
    probs = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-") or (tag.startswith('I-') and chunk[1] == -1):
            if chunk[2] != -1:
                _add_to_chunks(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                _add_to_chunks(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                _add_to_chunks(chunk)
        else:
            if chunk[2] != -1:
                _add_to_chunks(chunk)
            chunk = [-1, -1, -1]
    return [tuple(chunk) for chunk in chunks], probs
