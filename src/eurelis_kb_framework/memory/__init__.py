from langchain.schema import BaseMemory

from eurelis_kb_framework.base_factory import ProviderFactory


class GenericMemoryFactory(ProviderFactory[BaseMemory]):
    """
    Generic chains factory, provide with a conversational with memory question oriented chain
    """

    ALLOWED_PROVIDERS = {
        "conversation-buffer": "eurelis_kb_framework.memory.conversation_buffer_memory.ConversationBufferMemoryFactory",
    }

    def __init__(self):
        super().__init__()
        self.params["provider"] = "conversation-buffer"
