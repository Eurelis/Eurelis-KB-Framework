from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import BaseRetriever
from typing import TYPE_CHECKING

from eurelis_kb_framework.base_factory import ParamsDictFactory, DefaultFactories

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class SelfQueryRetrieverFactory(ParamsDictFactory[BaseRetriever]):
    def build(self, context: "BaseContext") -> BaseRetriever:
        llm = context.__class__.get_instance_from_factory(
            context, DefaultFactories.LLM, self.params.get("llm"), mandatory=True
        )

        if not llm:
            raise ValueError("Missing llm configuration for selfquery retriever")

        document_content_description = self.params.get("document_content_description")

        if not document_content_description:
            raise ValueError(
                "Missing document_content_description for selfquery retriever"
            )

        metadata_field_info = []
        for metadata_field in self.params.get("metadata_field_info"):
            name = metadata_field.get("name")
            description = metadata_field.get("description")
            field_type = metadata_field.get("type")

            metadata_field_info.append(
                AttributeInfo(name=name, description=description, type=field_type)
            )

        if not metadata_field_info:
            raise ValueError("Missing metadata_field_info for selfquery retriever")

        return SelfQueryRetriever.from_llm(
            llm,
            context.vector_store,
            document_content_description,
            metadata_field_info,
            chain_kwargs={"verbose": True},
        )
