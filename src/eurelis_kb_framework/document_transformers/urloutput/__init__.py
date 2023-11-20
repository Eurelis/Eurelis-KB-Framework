from typing import Sequence, Any

from langchain.schema import BaseDocumentTransformer, Document
from eurelis_kb_framework.base_factory import BaseFactory
from urllib.parse import urlparse, ParseResult


def strip_scheme(url):
    # https://stackoverflow.com/questions/21687408/how-to-remove-scheme-from-url-in-python
    parsed_result = urlparse(url)
    return ParseResult("", *parsed_result[1:]).geturl()


class UrlOutputTransformer(BaseDocumentTransformer):
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        for doc in documents:
            source = doc.metadata.get("source")
            stripped_source = strip_scheme(source)  # remove leading https/http/ftp...
            doc.metadata["source_output"] = stripped_source.strip(
                "/"
            )  # remove leading /

            yield doc

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError


class UrlOutputTransformerFactory(BaseFactory[BaseDocumentTransformer]):
    """
    Factory for the UrlOutputTransformer
    It is used to convert a metadata source value containing an url, to a source_output value containing a relative path
    Example,
    source = https://docs.python.org/3/library/exceptions.html
    => source_output = docs.python.org/3/library/exceptions.html

    It is mostly used for caching
    """

    def build(self, context) -> BaseDocumentTransformer:
        """
        Construct the UrlOutputTransformer
        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a UrlOutputTransformer instance
        """

        return UrlOutputTransformer()
