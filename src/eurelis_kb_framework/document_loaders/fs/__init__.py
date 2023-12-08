from typing import TYPE_CHECKING

from langchain.document_loaders.base import BaseLoader

from eurelis_kb_framework.base_factory import ParamsDictFactory
from eurelis_kb_framework.types import FACTORY

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class FSLoaderFactory(ParamsDictFactory[BaseLoader]):
    """
    File System Loader Factory
    """

    OPTIONAL_PARAMS = {"glob", "exclude", "suffixes", "show_progress"}

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.path = None
        self.parser_data = None

    def set_path(self, path: str):
        """
        Setter for the path parameter
        Args:
            path: the root path for looking to documents

        Returns:

        """
        self.path = path

    def set_parser(self, parser_data: FACTORY):
        """
        Setter for the parser factory data
        Args:
            parser_data: parser factory data

        Returns:

        """
        self.parser_data = parser_data

    def _process_parser_data(self, context, arguments: dict):
        """
        Helper method to add a 'parser' argument to a dict
        Args:
            context: the context object, usually the current langchain wrapper instance
            arguments: the dict to add the parser to

        Returns:

        """
        if self.parser_data and self.parser_data != "default":
            if isinstance(self.parser_data, str):
                parser_data = {"factory": self.parser_data}
            else:
                parser_data = self.parser_data.copy()

            parser_data.update({"path": self.path})

            parser = context.loader.instantiate_factory(
                "eurelis_kb_framework.parsers",
                "GenericBlobParserFactory",
                parser_data,
            )
            arguments["parser"] = parser.build(context)

    def build(self, context: "BaseContext") -> BaseLoader:
        """
        Construct the document loader

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            document loader

        """
        from langchain.document_loaders.generic import GenericLoader

        arguments = self.get_optional_params()
        self._process_parser_data(context, arguments)

        return GenericLoader.from_filesystem(self.path, **arguments)
