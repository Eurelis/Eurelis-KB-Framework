from langchain.chains.base import Chain

from eurelis_kb_framework import BaseFactory


class GenericChainsFactory(BaseFactory[Chain]):
    """
    Generic chains factory, provide with a conversational with memory question oriented cchain
    """

    def __init__(self):
        super().__init__()
        # self.system_template = """Use the following pieces of context to answer the users question.
        # Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES"
        # in capital letters regardless of the number of sources.
        # If you don't know the answer, just say that "I don't know", don't try to make up an answer.
        # ----------------
        # {summaries}"""
        self.retriever_kwargs = {}

    # def set_system_template(self, template: str):
    #    self.system_template = template

    def set_retriever_kwargs(self, kwargs: dict):
        """
        Setter for vector store kwargs
        Args:
            kwargs: key valye arguments to get the vector store retriever

        Returns:

        """
        self.retriever_kwargs = kwargs if kwargs else {}

    def build(self, context: "LangchainWrapper") -> Chain:
        """
        Construct the chain

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            langchain chain
        """
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory

        # memory object to persist call history between each calls
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer",
        )

        return ConversationalRetrievalChain.from_llm(
            context.lazy_get_llm(),
            retriever=context.vector_store.as_retriever(**self.retriever_kwargs),
            return_source_documents=True,
            memory=memory,
        )

        # from langchain.prompts.chat import (
        #     ChatPromptTemplate,
        #     SystemMessagePromptTemplate,
        #     HumanMessagePromptTemplate,
        # )
        #
        # messages = [
        #     SystemMessagePromptTemplate.from_template(self.system_template),
        #     HumanMessagePromptTemplate.from_template("{question}")
        # ]
        # prompt = ChatPromptTemplate.from_messages(messages)
        #
        # chain_type_kwargs = {"prompt": prompt}
        #
        # return RetrievalQAWithSourcesChain.from_chain_type(
        #    llm=context.lazy_get_llm(),
        #    chain_type="stuff",
        #    retriever=context.vector_store.as_retriever(**self.retriever_kwargs),
        #    return_source_documents=True,
        #    chain_type_kwargs=chain_type_kwargs
        # )
