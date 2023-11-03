from langchain.schema import BaseDocumentTransformer

from eurelis_kb_framework.base_factory import BaseFactory


class Html2TextTransformerFactory(BaseFactory[BaseDocumentTransformer]):
    """
    Factory for the Html2TextTransformer
    """

    def build(self, context) -> BaseDocumentTransformer:
        """
        Construct the Html2textTransformer
        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a Html2TextTransformer instance
        """
        from langchain.document_transformers.html2text import Html2TextTransformer

        return Html2TextTransformer()
