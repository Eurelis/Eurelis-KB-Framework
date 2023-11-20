from enum import Enum
from typing import List, Any, Dict, Optional
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage


class Method(Enum):
    MQAG = "mqag"
    BERTSCORE = "bertscore"
    NGRAM = "ngram"
    NLI = "nli"


class CheckInputCallback(BaseCallbackHandler):
    def __init__(self, checker, method: Method = Method.NLI, language: str = "en"):
        super().__init__()
        self.messages = []
        self.checker: ChatChecker = checker
        self.method = method
        self.language = language

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        self.messages = messages[
            0
        ]  # we have only one message in the chain at a given time

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None and "answer" in outputs:  # last chain_end
            answer = outputs.get("answer")

            values = self.checker.check(
                CheckInput(self.messages, answer, self.language), self.method
            )
            outputs.update({"selfcheck": values})


class CheckInput:
    def __init__(self, messages: List[BaseMessage], answer: str, language: str = "en"):
        self.messages = messages
        self.answer = answer

    @property
    def sentences(self):
        from nltk.tokenize import sent_tokenize

        return [sent.strip() for sent in sent_tokenize(self.answer)]


class ChatChecker:
    def __init__(self, chat_model: BaseChatModel):
        self.chat_model = chat_model

    def _produce_samples(
        self, messages: List[BaseMessage], samples: int = 4
    ) -> list[str]:
        return [self.chat_model(messages).content for _ in range(samples)]

    def check(self, input: CheckInput, method: Method = Method.NLI) -> dict[str, Any]:
        import torch
        from selfcheckgpt.modeling_selfcheck import (
            # SelfCheckMQAG,
            SelfCheckBERTScore,
            SelfCheckNgram,
            SelfCheckNLI,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # selfcheck_mqag = SelfCheckMQAG(
        #    device=device
        # )  # set device to 'cuda' if GPU is available

        samples = self._produce_samples(input.messages)

        sentences = input.sentences

        results = {}
        if not method or method == Method.BERTSCORE:
            selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
            results[Method.BERTSCORE.value] = selfcheck_bertscore.predict(
                sentences=sentences, sampled_passages=samples
            )
        if not method or method == Method.NLI:
            selfcheck_nli = SelfCheckNLI(
                device=device
            )  # set device to 'cuda' if GPU is available
            results[Method.NLI.value] = selfcheck_nli.predict(
                sentences=sentences,  # list of sentences
                sampled_passages=samples,  # list of sampled passages
            )
        if not method or method == Method.NGRAM:
            selfcheck_ngram = SelfCheckNgram(
                n=1
            )  # n=1 means Unigram, n=2 means Bigram, etc.
            results[Method.NGRAM.value] = selfcheck_ngram.predict(
                sentences=sentences,
                passage=input.answer,
                sampled_passages=samples,
            )

        return results
