import os
from itertools import islice
from string import Template


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
    batch = tuple(islice(it, n))
    while batch:
        yield batch
        batch = tuple(islice(it, n))


def parse_param_value(raw_value: str) -> str:
    """
    Method to handle parameter values, will resolve environment variable values if needed

    Args:
        raw_value (str): the raw parameter value from the configuration file

    Returns:
        parameter value (str), the final value to use

    Raise:
        ValueError: If an environment variable value isn't found

    """
    if raw_value and isinstance(raw_value, str) and "$" in raw_value:
        # if the value start with an $, it is assumed to be the name of an environment variable
        # use $$ prefix to escape this mode
        s = Template(raw_value)

        if not s.is_valid():
            raise ValueError(f"Invalid {raw_value} pattern")

        return s.substitute(os.environ)

    return raw_value
