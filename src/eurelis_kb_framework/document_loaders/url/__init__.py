from typing import TYPE_CHECKING

from langchain.document_loaders.base import BaseLoader

from eurelis_kb_framework.base_factory import ParamsDictFactory

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class UrlLoaderFactory(ParamsDictFactory[BaseLoader]):
    """
    Url loader factory
    """

    OPTIONAL_PARAMS = {
        "use_async",
        "exclude_dirs",
        "timeout",
        "prevent_outside",
        "link_regex",
        "headers",
        "check_response_status",
    }

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.is_recursive = False
        self.url = None

    def set_url(self, url: str):
        """
        Setter for the url parameter
        Args:
            url: url value

        Returns:

        """
        self.url = url

    def set_recursive(self, recursive: bool):
        """
        Setter for the recursive parameter
        Args:
            recursive: boolean, should we use a recursive approach? default to False

        Returns:

        """
        self.is_recursive = recursive

    def set_max_depth(self, max_depth: int):
        """
        Setter for the max_depth parameter
        Args:
            max_depth: integer, how deep should we follow links, default to 1: non recursive

        Returns:

        """
        self.params["max_depth"] = max(1, max_depth)
        if max_depth > 1:
            self.is_recursive = True

    def build(self, context: "BaseContext") -> BaseLoader:
        """
        Construct the url document loader

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a document loader
        """
        from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader

        if not self.is_recursive:
            self.params["max_depth"] = 1

        return RecursiveUrlLoader(self.url, **self.params)
