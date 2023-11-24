import types
from typing import Any

from langchain.document_loaders.base import BaseLoader

from eurelis_kb_framework.base_factory import ParamsDictFactory

default_header_template = {
    "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
    # ";q=0.8",
    # "Accept-Language": "en-US,en;q=0.5",
    # "Referer": "https://www.google.com/",
    # "DNT": "1",
    # "Connection": "keep-alive",
    # "Upgrade-Insecure-Requests": "1",
}


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

    if title := content.find("title"):
        metadata["title"] = title.get_text()
    if description := content.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", "")
    if html := content.find("html"):
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

    def build(self, context) -> BaseLoader:
        """
        Construct the sitemap document loader

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a document loader
        """
        from langchain.document_loaders.sitemap import SitemapLoader

        web_path = self.params.get("web_path")

        if not web_path:
            raise ValueError("Missing required web_path parameter")

        parameters = self.get_optional_params()

        loader = SitemapLoader(
            web_path,
            parsing_function=_parsing_function_factory(
                self.params.get("parser_remove")
            ),
            meta_function=_meta_function,
            header_template=default_header_template.copy(),
            **parameters,
        )

        # hack as SitemapLoader does not override lazy_load from WebBaseLoader, we will force
        # the dataset to use the load method instead
        def lazy_load(self):
            raise NotImplementedError("")

        loader.lazy_load = types.MethodType(lazy_load, loader)

        return loader
