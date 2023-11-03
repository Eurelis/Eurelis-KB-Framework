from itertools import islice


def batched(iterable, n):
    """
    Batch data into tuples of length n. The last batch may be shorter.
    Args:
        iterable: source for the chunks
        n: size of the chunks, last one can be smaller

    Returns:
        iterator for chunks

    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
