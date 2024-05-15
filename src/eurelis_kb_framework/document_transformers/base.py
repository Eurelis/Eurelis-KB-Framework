from abc import ABC, abstractmethod
from typing import Iterable, Any

from langchain_core.documents import Document
from langchain_core.runnables import run_in_executor


class BaseIteratorDocumentTransformer(ABC):
    """
    Based on langchain BaseIteratorTransformer, but using iterable instead of sequence

    Abstract base class for document transformation systems.

    A document transformation system takes an iterable of Documents and returns an iterable of transformed Documents.


    """  # noqa: E501

    @abstractmethod
    def transform_documents(
        self, documents: Iterable[Document], **kwargs: Any
    ) -> Iterable[Document]:
        """Transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """

    async def atransform_documents(
        self, documents: Iterable[Document], **kwargs: Any
    ) -> Iterable[Document]:
        """Asynchronously transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """
        return await run_in_executor(
            None, self.transform_documents, documents, **kwargs
        )
