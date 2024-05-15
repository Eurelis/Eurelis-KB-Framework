from typing import Union, Any, TYPE_CHECKING

from langchain.chains.base import Chain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseMemory

from eurelis_kb_framework.base_factory import (
    ParamsDictFactory,
    DefaultFactories,
)
from eurelis_kb_framework.types import FACTORY

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


def _extract_list_or_string(value: Any) -> Any:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return " ".join(value)
    elif not isinstance(value, str):
        raise ValueError(
            f"Unexpected value given to _extract_list_or_string, expected single string or list of string got {type(value)}"
        )

    return value


class ConversationalRetrievalChainFactory(ParamsDictFactory[Chain]):
    """
    Factory for the conversational retrieval chain
    """

    def __init__(self):
        super().__init__()
        self.retriever_kwargs = {}
        self.memory = None

        self.condense_question_prompt = None

        self.combine_docs_chain_kwargs = None
        self.condense_question_llm_factory = None
        self.output_format = None
        self.output_field = None

    def set_condense_question_llm(self, value: FACTORY):
        if isinstance(value, dict):
            self.condense_question_llm_factory = value.copy()
        elif isinstance(value, str):
            self.condense_question_llm_factory = value
        else:
            raise ValueError(
                f"condense_question_llm parameter is expected to be a factory (str ou dict), got {value} {type(value)}"
            )

    def set_condense_question_prompt(self, value: Union[str, list]):
        value = _extract_list_or_string(value)
        if not isinstance(value, str):
            raise ValueError(
                "Bad condensed_question_prompt value given, expected str got {type(str)}"
            )
        if not value or "{question}" not in value or "{chat_history}" not in value:
            raise ValueError(
                "Bad condensed_question_prompt value, expecting a string containing {chat_history}, {question}"
            )

        self.condense_question_prompt = PromptTemplate.from_template(value)

    def set_combine_docs_chain_kwargs(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError(
                f"Bad combine_docs_chain_kwargs value, expecting a dict, got {type(value)}"
            )

        self.combine_docs_chain_kwargs = value.copy()
        if "prompt" in value:
            prompt_value = value.get("prompt")
            if not isinstance(prompt_value, dict):
                raise ValueError(
                    f"Bad combine_docs_chain_kwargs prompt value, expecting a dict, got {type(value)}"
                )
            system = _extract_list_or_string(prompt_value.get("system"))
            human = _extract_list_or_string(prompt_value.get("human"))

            if not system or not isinstance(system, str) or "{context}" not in system:
                raise ValueError(
                    "Bad combine_docs_chain_kwargs prompt value, "
                    + "should have contain a system key with an associated text (or a list) containing {context}"
                )
            if not human or not isinstance(human, str) or "{question}" not in human:
                raise ValueError(
                    "Bad combine_docs_chain_kwargs prompt value, "
                    + "should have contain a human key with an associated text (or a list) containing {question}"
                )

            self.combine_docs_chain_kwargs["prompt"] = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(system),
                    HumanMessagePromptTemplate.from_template(human),
                ]
            )

    def set_memory(self, memory: Union[BaseMemory, FACTORY]):
        """
        Setter for the memory object to use
        Args:
            memory: instance of BaseMemory object or
        """
        self.memory = memory

    def build(self, context: "BaseContext") -> Chain:
        """
        Construct the chain

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            langchain chain
        """
        from langchain.chains import ConversationalRetrievalChain
        from eurelis_kb_framework.langchain_wrapper import LangchainWrapper

        if not isinstance(context, LangchainWrapper):
            raise RuntimeError(
                "ConversationalRetrievalChain must be used with a LangchainWrapper instance as context"
            )

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

        other_args = {}
        if self.condense_question_prompt:
            other_args["condense_question_prompt"] = self.condense_question_prompt
        if self.combine_docs_chain_kwargs:
            other_args["combine_docs_chain_kwargs"] = self.combine_docs_chain_kwargs
        if self.condense_question_llm_factory:
            other_args[
                "condense_question_llm"
            ] = context.__class__.get_instance_from_factory(
                context,
                DefaultFactories.LLM,
                self.condense_question_llm_factory,
                mandatory=True,
            )

        # build and return the chain
        return ConversationalRetrievalChain.from_llm(
            context.lazy_get_llm(),
            retriever=retriever,
            return_source_documents=True,
            memory=memory,
            **other_args,
        )
