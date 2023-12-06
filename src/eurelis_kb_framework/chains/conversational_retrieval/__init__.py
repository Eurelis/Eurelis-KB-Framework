from typing import Union

from langchain.chains.base import Chain
from langchain.schema import BaseMemory

from eurelis_kb_framework.base_factory import (
    ParamsDictFactory,
    DefaultFactories,
)
from eurelis_kb_framework.types import FACTORY


class ConversationalRetrievalChainFactory(ParamsDictFactory[Chain]):

    """
    Factory for the conversational retrieval chain
    """

    def __init__(self):
        super().__init__()
        self.retriever_kwargs = {}
        self.memory = None

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

        # we check the type for the memory value
        if self.memory is None:
            # build from default factory
            memory = context.__class__.get_instance_from_factory(
                context, DefaultFactories.MEMORY, {}, mandatory=True
            )
        elif isinstance(self.memory, BaseMemory):
            # use as it
            memory = self.memory
        elif isinstance(self.memory, (str, dict)):
            # build from factory
            memory = context.__class__.get_instance_from_factory(
                context, DefaultFactories.MEMORY, self.memory
            )

        retriever = context.__class__.get_instance_from_factory(
            context,
            DefaultFactories.RETRIEVER,
            self.params.get("retriever", dict()),
            mandatory=True,
        )

        # build and return the chain
        return ConversationalRetrievalChain.from_llm(
            context.lazy_get_llm(),
            retriever=retriever,
            return_source_documents=True,
            memory=memory,
        )
