from typing import Optional, Dict, List, Tuple, Callable, Any

from bson import ObjectId
from langchain.schema import Document
from langchain_community.vectorstores import MongoDBAtlasVectorSearch


class MongoDBSimilarityAtlasVectorStoreSearch(MongoDBAtlasVectorSearch):
    """
    Class to enable similarity search with score on mongodb
    """

    # def similarity_search_with_score(
    #     self,
    #     query: str,
    #     k: int = 4,
    #     *,
    #     pre_filter: Optional[Dict] = None,
    #     post_filter_pipeline: Optional[List[Dict]] = None,
    # ) -> List[Tuple[Document, float]]:
    #     """Override to fix an issue
    #     TypeError: MongoDBAtlasVectorSearch.similarity_search_with_score() takes 2 positional arguments but 3 were given
    #
    #     Args:
    #         query:
    #         k:
    #         pre_filter:
    #         post_filter_pipeline:
    #
    #     Returns:
    #
    #     """
    #     return super().similarity_search_with_score(
    #         query, k=k, pre_filter=pre_filter, post_filter_pipeline=post_filter_pipeline
    #     )

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]: Documents to add to the vectorstore.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        if "ids" in kwargs:
            # to enable deleting documents we must store the unique id used by the index system
            ids = kwargs["ids"]
            if len(ids) != len(documents):
                raise ValueError("ids length mismatch documents length")

            for doc_index, doc in enumerate(documents):
                doc.metadata["_uid"] = ids[doc_index]

        # TODO: Handle the case where the user doesn't provide ids on the Collection
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """

        if not ids:
            return True

        del_query = {"_uid": {"$in": ids}}
        self._collection.delete_many(del_query)

        return True
