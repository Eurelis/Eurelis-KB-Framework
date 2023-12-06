from langchain.schema import BaseRetriever

from eurelis_kb_framework.base_factory import ProviderFactory


class GenericRetrieverFactory(ProviderFactory[BaseRetriever]):
    ALLOWED_PROVIDERS = {
        "vectorstore": "eurelis_kb_framework.retrievers.vectorstore.VectorStoreRetrieverFactory",
        "selfcheck": "eurelis_kb_framework.retrievers.selfquery.SelfQueryRetrieverFactory",
    }

    def __init__(self):
        super().__init__()
        self.params["provider"] = "vectorstore"
