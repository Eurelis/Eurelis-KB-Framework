from enum import Enum

from eurelis_kb_framework.addons.checker.check_input import CheckInput


class Method(Enum):
    MQAG = "mqag"
    BERTSCORE = "bertscore"
    NGRAM = "ngram"
    NLI = "nli"
