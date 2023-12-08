import types
from typing import Any, TYPE_CHECKING

from langchain.document_loaders.base import BaseLoader

from eurelis_kb_framework.base_factory import ParamsDictFactory

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


def _parsing_function_factory(remove_list: list):
    if not remove_list:
        return lambda content: content.get_text()

    def _parsing_function(content: Any) -> str:
        for remove in remove_list:
            if isinstance(remove, str):  # tagname
                nodes = content.find_all(remove)
                for node in nodes:
                    node.extract()
            elif isinstance(remove, dict):
                nodes = content.find_all(**remove)
                for node in nodes:
                    node.extract()

        return content.get_text("\n")

    return _parsing_function


def _meta_function(meta: dict, content: Any) -> dict:
    metadata = {"source": meta["loc"], **meta}

    title = content.find("title")
    description = content.find("meta", attrs={"name": "description"})
    html = content.find("html")

    if title:
        metadata["title"] = title.get_text()
    if description:
        metadata["description"] = description.get("content", "")
    if html:
        metadata["language"] = html.get("lang", "")

    return metadata


class SitemapDocumentLoaderFactory(ParamsDictFactory[BaseLoader]):
    OPTIONAL_PARAMS = {
        "filter_urls",
        "blocksize",
        "is_local",
        "continue_on_failure",
        "restrict_to_same_domain",
    }

    def build(self, context: "BaseContext") -> BaseLoader:
        """
        Construct the sitemap document loader

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a document loader
        """
        from langchain.document_loaders.sitemap import SitemapLoader
        from langchain.document_loaders.web_base import default_header_template

        web_path = self.params.get("web_path")

        if not web_path:
            raise ValueError("Missing required web_path parameter")

        header_template = default_header_template.copy()
        # default user agent is "EurelisKBF/0.1"
        header_template["User-Agent"] = self.params.get("user_agent", "EurelisKBF/0.1")

        parameters = self.get_optional_params()

        loader = SitemapLoader(
            web_path,
            parsing_function=_parsing_function_factory(
                self.params.get("parser_remove")
            ),
            meta_function=_meta_function,
            header_template=header_template,
            **parameters,
        )

        # hack as SitemapLoader does not override lazy_load from WebBaseLoader, we will force
        # the dataset to use the load method instead
        def lazy_load(instance):
            raise NotImplementedError("")

        loader.lazy_load = types.MethodType(lazy_load, loader)

        return loader
