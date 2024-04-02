from typing import TYPE_CHECKING

from langchain.llms.base import BaseLLM
from langchain_core.language_models import BaseChatModel

from eurelis_kb_framework.base_factory import ProviderFactory, ParamsDictFactory

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class ChatOpenAIFactory(ParamsDictFactory[BaseLLM]):
    OPTIONAL_PARAMS = {
        "cache",
        "max_retries",
        "max_tokens",
        "metadata",
        "model_kwargs",
        "model_name",
        "n",
        "openai_ai_base",
        "openai_api_key",
        "openai_organization",
        "openai_proxy",
        "request_timeout",
        "streaming",
        "tags",
        "temperature",
        "tiktoken_model_name",
        "verbose",
    }

    def build(self, context: "BaseContext") -> BaseChatModel:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore

            arguments = self.get_optional_params()

            return ChatOpenAI(**arguments)  # type: ignore

        except ImportError:
            raise ImportError(
                "Please install langchain_openai with the option openai, (pip install eurelis_kb_framework[openai]"
            )


class OpenAIFactory(ParamsDictFactory[BaseLLM]):
    OPTIONAL_PARAMS = {
        "allowed_special",
        "batch_size",
        "best_of",
        "cache",
        "disallowed_special",
        "frequency_penalty",
        "logit_bias",
        "max_retries",
        "max_tokens",
        "metadata",
        "model_kwargs",
        "model_name",
        "n",
        "openai_api_base",
        "openai_api_key",
        "openai_organization",
        "openai_proxy",
        "presence_penalty",
        "request_timeout",
        "streaming",
        "tags",
        "temperature",
        "tiktoken_model_name",
        "top_p",
        "verbose",
    }

    def build(self, context: "BaseContext") -> BaseLLM:
        try:
            from langchain_openai import OpenAI  # type: ignore

            arguments = self.get_optional_params()

            return OpenAI(**arguments)  # type: ignore[arg-type]

        except ImportError:
            raise ImportError(
                "Please install langchain_openai with the option openai, (pip install eurelis_kb_framework[openai]"
            )


class GenericOpenAIFactory(ProviderFactory[BaseLLM]):
    ALLOWED_PROVIDERS = {"chat-openai": ChatOpenAIFactory, "openai": OpenAIFactory}
