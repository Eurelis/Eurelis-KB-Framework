from typing import TYPE_CHECKING

from langchain.schema.vectorstore import VectorStore

from eurelis_kb_framework.base_factory import ParamsDictFactory

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class MongoDBVectorStoreFactory(ParamsDictFactory[VectorStore]):
    """
    Factory to get a mongodb based vector store
    """

    def build(self, context: "BaseContext") -> VectorStore:
        """
        Construct a chromadb based vector store

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a chroma vector store object
        """
        from pymongo import MongoClient
        from langchain.vectorstores import MongoDBAtlasVectorSearch

        url = self.params.get("url")
        db_name = self.params.get("db_name", "knowledge_base")
        collection_name = self.params.get("collection_name", context.project)
        if not url:
            raise ValueError(
                "Missing required URL parameter for MongoDBVectorStoreFactory"
            )

        context.console.verbose_print(
            f"Getting MongoDB vector store, using database {db_name} and collection {collection_name}"
        )

        mongo_client = MongoClient(url)
        collection = mongo_client[db_name][collection_name]

        return MongoDBAtlasVectorSearch(collection, context.embeddings)
