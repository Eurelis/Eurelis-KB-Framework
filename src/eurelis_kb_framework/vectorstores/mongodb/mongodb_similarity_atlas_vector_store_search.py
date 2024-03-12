from typing import Optional, List, Any, Callable

from langchain.schema import Document
from langchain_community.vectorstores import MongoDBAtlasVectorSearch


class MongoDBSimilarityAtlasVectorStoreSearch(MongoDBAtlasVectorSearch):
    """
    Class to enable similarity search with score on mongodb
    """

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        Vectorstores should define their own selection based method of relevance.
        """
        # override default implementation which is buggy
        return lambda x: x

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
