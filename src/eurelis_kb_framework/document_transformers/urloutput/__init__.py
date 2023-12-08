from typing import Sequence, Any, TYPE_CHECKING
from urllib.parse import urlparse, ParseResult

from langchain.schema import BaseDocumentTransformer, Document

from eurelis_kb_framework.base_factory import BaseFactory

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


def _strip_scheme(url: str) -> str:
    """Function to remove the scheme from an url

    Parse an url then return it removing its scheme and leading "/" characcters
    ie: https://apple.com => //apple.com => apple.com

    Args:
        url (str): an url as a string

    Returns:
        str: the url without its scheme

    """
    # https://stackoverflow.com/questions/21687408/how-to-remove-scheme-from-url-in-python
    parsed_result = urlparse(url)  # https://apple.com
    without_scheme = ParseResult("", *parsed_result[1:]).geturl()  # //apple.com

    return without_scheme.strip("/")  # apple.com


class UrlOutputTransformer(BaseDocumentTransformer):
    def __init__(
        self, url_source_field: str = "source", path_output_field: str = "source_output"
    ):
        self._url_source_field = url_source_field
        self._path_output_field = path_output_field

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Add a new metadata field containing a relative path given a source url.

        This method will extract the url from the document metadata, then remove its scheme and add it back to metadata
        in a new field.

        Args:
            documents: A sequence of documents to be transformed.
            **kwargs: Arbitrary keyword arguments.

        Yields:
            Sequence[Document]: sequence of transformed documents

        """

        for doc in documents:
            source_url = doc.metadata.get(self._url_source_field)
            doc.metadata[self._path_output_field] = _strip_scheme(
                source_url
            )  # remove leading https/http/ftp...

            yield doc

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously add a new metadata field containing a relative path given a source url.

        This method isn't implemented

        Raises:
            NotImplementedError
        """

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

    def build(self, context: "BaseContext") -> BaseDocumentTransformer:
        """
        Construct the UrlOutputTransformer
        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a UrlOutputTransformer instance
        """

        return UrlOutputTransformer()
