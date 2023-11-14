from langchain.document_loaders.base import BaseLoader

from eurelis_kb_framework.base_factory import ProviderFactory


class GenericLoaderFactory(ProviderFactory[BaseLoader]):
    """
    Generic loader factory
    """

    ALLOWED_PROVIDERS = {
        "url": "eurelis_kb_framework.document_loaders.url.UrlLoaderFactory",
        "fs": "eurelis_kb_framework.document_loaders.fs.FSLoaderFactory",
        "list": "eurelis_kb_framework.document_loaders.list.ListLoaderFactory",
        "sitemap": "eurelis_kb_framework.document_loaders.sitemap.SitemapDocumentLoaderFactory",
    }
