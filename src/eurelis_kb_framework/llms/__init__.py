from langchain.llms.base import BaseLLM

from eurelis_kb_framework.base_factory import ProviderFactory
from eurelis_kb_framework.llms.openai import GenericOpenAIFactory


class GenericLLMFactory(ProviderFactory[BaseLLM]):
    ALLOWED_PROVIDERS = {
        **GenericOpenAIFactory.ALLOWED_PROVIDERS,
        "huggingface-pipeline": "eurelis_kb_framework.llms.huggingface.HuggingFaceFactory",
    }
