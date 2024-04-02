from typing import TYPE_CHECKING, cast

from langchain.schema.vectorstore import VectorStore

from eurelis_kb_framework.base_factory import ParamsDictFactory

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


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

    def build(self, context: "BaseContext") -> VectorStore:
        """
        Construct a mongodb based vector store

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a mongodb vector store object
        """
        from eurelis_kb_framework.langchain_wrapper import LangchainWrapper

        if not isinstance(context, LangchainWrapper):
            raise RuntimeError(
                "MongoDBVectorStoreFactory expects a LangchainWrapper as build context"
            )

        from pymongo import MongoClient  # type: ignore[import-not-found]
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

        #
        # FIXME
        # VLA - le 07/03/2024
        # This part of the code is commented because it is not working
        # An error is raised when trying to create the search index
        # pymongo.errors.OperationFailure: command not found, full error: {'ok': 0, 'errmsg': 'command not found', 'code': 59, 'codeName': 'CommandNotFound'}
        #
        server_info = mongo_client.server_info()
        print(server_info)
        version_array = server_info["versionArray"]

        search_index = {
            "name": cast(str, other_params.get("index_name")),
            "definition": {
                "type": "vectorSearch",
                "fields": [
                    {
                        "numDimensions": len(context.embeddings.embed_query("")),
                        "path": cast(str, other_params.get("embedding_key")),
                        "similarity": "cosine",
                        "type": "vector",
                    }
                ],
            },
        }

        display_search_index_info = True

        if version_array[0] >= 7:
            from pymongo.errors import OperationFailure

            try:
                collection.create_search_index(search_index)
                display_search_index_info = False
            except OperationFailure:
                context.console.print("Unable to auto create search index")

        if display_search_index_info:
            context.console.print("MongoDB vector store search index configuration:")
            import json

            context.console.print(json.dumps(search_index))

        return MongoDBSimilarityAtlasVectorStoreSearch(
            collection, context.embeddings, **other_params
        )
