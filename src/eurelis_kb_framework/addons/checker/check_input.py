from typing import List

from langchain.schema import BaseMessage


class CheckInput:
    def __init__(self, messages: List[BaseMessage], answer: str, language: str = "en"):
        self.messages = messages
        self.answer = answer

    @property
    def sentences(self):
        from nltk.tokenize import sent_tokenize  # type: ignore[import-not-found]

        return [sent.strip() for sent in sent_tokenize(self.answer)]
