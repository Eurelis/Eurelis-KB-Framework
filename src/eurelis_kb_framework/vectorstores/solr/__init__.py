from langchain.schema.vectorstore import VectorStore

from eurelis_kb_framework.base_factory import BaseFactory


class SolrFactory(BaseFactory[VectorStore]):
    """
    Factory to get a solr based vector store
    """

    def build(self, context) -> VectorStore:
        """
        Construct a solr based vector store

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a Solr vector store object
        """
        from eurelis_langchain_solr_vectorstore import Solr

        context.console.verbose_print(f"Getting solr vector store")

        return Solr(context.embeddings)
