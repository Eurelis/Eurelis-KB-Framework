from langchain.schema.vectorstore import VectorStore

from eurelis_kb_framework.base_factory import ProviderFactory


class GenericVectorStoreFactory(ProviderFactory[VectorStore]):
    """
    Generic factory for vector store, delegate to another factory under the hood
    """

    ALLOWED_PROVIDERS = {
        "chroma": "eurelis_kb_framework.vectorstores.chroma.ChromaFactory",
        "solr": "eurelis_kb_framework.vectorstores.solr.SolrFactory",
        "mongodb": "eurelis_kb_framework.vectorstores.mongodb.MongoDBVectorStoreFactory",
    }
