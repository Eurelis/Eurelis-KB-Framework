from typing import TYPE_CHECKING

from langchain.schema.vectorstore import VectorStore

from eurelis_kb_framework.base_factory import ParamsDictFactory

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import LangchainWrapper


class MongoDBVectorStoreFactory(ParamsDictFactory[VectorStore]):
    """
    Factory to get a mongodb based vector store
    """

    OPTIONAL_PARAMS = {"index_name", "text_key", "embedding_key"}

    def __init__(self):
        super().__init__()
        self.params["index_name"] = "default"
        self.params["text_key"] = "text"
        self.params["embedding_key"] = "embedding"

    def build(self, context: "LangchainWrapper") -> VectorStore:
        """
        Construct a mongodb based vector store

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a mongodb vector store object
        """
        from pymongo import MongoClient
        from eurelis_kb_framework.vectorstores.mongodb.mongodb_similarity_atlas_vector_store_search import (
            MongoDBSimilarityAtlasVectorStoreSearch,
        )

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

        other_params = self.get_optional_params()

        version_array = mongo_client.server_info()["versionArray"]
        if version_array[0] >= 7:
            search_index = {
                "name": other_params.get("index_name"),
                "definition": {
                    "type": "vectorSearch",
                    "fields": [
                        {
                            "numDimensions": len(context.embeddings.embed_query("")),
                            "path": other_params.get("embedding_key"),
                            "similarity": "cosine",
                            "type": "vector",
                        }
                    ],
                },
            }

            collection.create_search_index(search_index)

        return MongoDBSimilarityAtlasVectorStoreSearch(
            collection, context.embeddings, **other_params
        )
