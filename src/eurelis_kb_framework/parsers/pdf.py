from typing import Iterator, TYPE_CHECKING
import os

from langchain.document_loaders import Blob
from langchain.document_loaders.base import BaseBlobParser
from langchain.schema import Document

from eurelis_kb_framework.base_factory import ParamsDictFactory

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class PdfFileParserFactory(ParamsDictFactory[BaseBlobParser]):
    def build(self, context: "BaseContext") -> BaseBlobParser:
        return PdfFileParser(self.params.get("path", "/"))


class PdfFileParser(BaseBlobParser):
    def __init__(self, base_path: str):
        self._base_path = base_path

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """

        :param blob:
        :return:
        """
        from pypdf import PdfReader

        metadata = {
            "source": str(os.path.relpath(blob.path, self._base_path)),
        }

        pdf_file = PdfReader(blob.path)

        metadata.update(pdf_file.metadata)

        content = ""
        for page in pdf_file.pages:
            content += f"{page.extract_text()}\n\n"

        yield Document(page_content=content, metadata=metadata)
