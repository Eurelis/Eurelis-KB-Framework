from typing import Optional, Sequence, Any

from langchain.schema import BaseDocumentTransformer, Document

from eurelis_kb_framework.acronyms import AcronymsTextTransformer


class AcronymsDocumentTransformer(BaseDocumentTransformer):
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        if self.chain:
            # if there is a chain we yield from it, otherwise we directly yield the document (more efficient this way than doing a check in the loop)
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
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError

    def __init__(
        self,
        acronyms: AcronymsTextTransformer,
        chain_transformer: Optional[BaseDocumentTransformer] = None,
    ):
        self.acronyms = acronyms
        self.chain = chain_transformer
