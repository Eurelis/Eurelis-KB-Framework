from typing import TYPE_CHECKING

from langchain.schema.embeddings import Embeddings

from eurelis_kb_framework.base_factory import ParamsDictFactory

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class HuggingFaceEmbeddingsFactory(ParamsDictFactory[Embeddings]):
    OPTIONAL_PARAMS = {"cache_folder", "encode_kwargs", "model_name", "multi_process"}

    def build(self, context: "BaseContext") -> Embeddings:
        """
        Construct the embeddings object

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            embeddings
        """
        from langchain.embeddings import HuggingFaceEmbeddings

        arguments = self.get_optional_params()

        return HuggingFaceEmbeddings(**arguments)
