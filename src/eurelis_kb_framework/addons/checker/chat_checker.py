from typing import List, Any, Optional

from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage

from eurelis_kb_framework.addons.checker import Method, CheckInput


class ChatChecker:
    def __init__(self, chat_model: BaseChatModel):
        self.chat_model = chat_model

    def _produce_samples(
        self, messages: List[BaseMessage], samples: int = 4
    ) -> list[str]:
        return [self.chat_model(messages).content for _ in range(samples)]

    def check(
        self, input: CheckInput, method: Optional[Method] = Method.NLI
    ) -> dict[str, Any]:
        if method == Method.MQAG:
            raise NotImplementedError

        import torch  # type: ignore[import-not-found]
        from selfcheckgpt.modeling_selfcheck import (  # type: ignore[import-not-found]
            # SelfCheckMQAG,
            SelfCheckBERTScore,
            SelfCheckNgram,
            SelfCheckNLI,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        samples = self._produce_samples(input.messages)

        sentences = input.sentences

        results = {}
        if not method or method == Method.BERTSCORE:
            bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
            results[Method.BERTSCORE.value] = bertscore.predict(
                sentences=sentences, sampled_passages=samples
            )
        if not method or method == Method.NLI:
            nli = SelfCheckNLI(
                device=device
            )  # set device to 'cuda' if GPU is available
            results[Method.NLI.value] = nli.predict(
                sentences=sentences,  # list of sentences
                sampled_passages=samples,  # list of sampled passages
            )
        if not method or method == Method.NGRAM:
            ngram = SelfCheckNgram(n=1)  # n=1 means Unigram, n=2 means Bigram, etc.
            results[Method.NGRAM.value] = ngram.predict(
                sentences=sentences,
                passage=input.answer,
                sampled_passages=samples,
            )

        return results
