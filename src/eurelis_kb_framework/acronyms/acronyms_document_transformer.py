from typing import Optional, Sequence, Any, Union, Iterable

from langchain.schema import BaseDocumentTransformer, Document

from eurelis_kb_framework.acronyms import AcronymsTextTransformer
from eurelis_kb_framework.document_transformers.base import (
    BaseIteratorDocumentTransformer,
)


class AcronymsDocumentTransformer(BaseIteratorDocumentTransformer):
    """
    Acronyms document transformer, document transformer performing an acronyms transformation
    """

    def __init__(
        self,
        acronyms: AcronymsTextTransformer,
        chain_transformer: Optional[
            Union[BaseDocumentTransformer, BaseIteratorDocumentTransformer]
        ] = None,
    ):
        """
        Initializer

        Args:
            acronyms (AcronymsTextTransformer): the acronyms transformer
            chain_transformer: optional transformer to call after acronyms transformation
        """

        self.acronyms = acronyms
        self.chain = chain_transformer

    def transform_documents(
        self, documents: Iterable[Document], **kwargs: Any
    ) -> Iterable[Document]:
        """
        Transform documents implementation
        """
        # if there is a chain we yield from it, otherwise we directly yield the document (more efficient this way than doing a check in the loop)

        if self.chain:
            for doc in documents:
                new_doc = Document(
                    page_content=self.acronyms.transform(doc.page_content),
                    metadata=doc.metadata.copy(),
                )

                yield from self.chain.transform_documents([new_doc])
        else:
            for doc in documents:
                new_doc = Document(
                    page_content=self.acronyms.transform(doc.page_content),
                    metadata=doc.metadata.copy(),
                )
                yield new_doc

    async def atransform_documents(
        self, documents: Iterable[Document], **kwargs: Any
    ) -> Iterable[Document]:
        raise NotImplementedError
