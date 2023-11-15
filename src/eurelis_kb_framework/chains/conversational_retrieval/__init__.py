from typing import Union

from langchain.chains.base import Chain
from langchain.schema import BaseMemory

from eurelis_kb_framework.base_factory import (
    ParamsDictFactory,
    FACTORY,
    DefaultFactories,
)


class ConversationalRetrievalChainFactory(ParamsDictFactory[Chain]):
    def __init__(self):
        super().__init__()
        self.retriever_kwargs = {}
        self.memory = None

    def set_retriever_kwargs(self, kwargs: dict):
        """
        Setter for vector store kwargs
        Args:
            kwargs: key value arguments to get the vector store retriever

        Returns:

        """
        self.retriever_kwargs = kwargs if kwargs else {}

    def set_memory(self, memory: Union[BaseMemory, FACTORY]):
        """
        Setter for the memory object to use
        Args:
            memory: instance of BaseMemory object or
        """
        self.memory = memory

    def build(self, context: "LangchainWrapper") -> Chain:
        """
        Construct the chain

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            langchain chain
        """
        from langchain.chains import ConversationalRetrievalChain

        memory = None
        if self.memory is None:
            memory = context.__class__.get_instance_from_factory(
                context, DefaultFactories.MEMORY, {}, mandatory=True
            )
        elif isinstance(self.memory, BaseMemory):
            memory = self.memory
        elif isinstance(self.memory, (str, dict)):
            memory = context.__class__.get_instance_from_factory(
                context, DefaultFactories.MEMORY, self.memory
            )

        return ConversationalRetrievalChain.from_llm(
            context.lazy_get_llm(),
            retriever=context.vector_store.as_retriever(**self.retriever_kwargs),
            return_source_documents=True,
            memory=memory,
        )
