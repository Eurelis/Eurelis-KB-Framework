from typing import TYPE_CHECKING

import chromadb
from chromadb import API
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import Chroma

from eurelis_kb_framework.base_factory import BaseFactory

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class ChromaFactory(BaseFactory[VectorStore]):
    """
    Factory to get a chroma based vector store
    """

    ALLOWED_MODES = {"in-memory", "persistent", "http"}

    def __init__(self):
        self.arguments = {}
        self.mode = "in-memory"
        self.path = None
        self.host = "localhost"
        self.port = 8000

    def set_collection_name(self, name: str):
        """
        Setter for the collection name
        Args:
            name: collection name

        Returns:

        """
        self.arguments["collection_name"] = name

    def set_mode(self, mode: str):
        """
        Setter for the access method
        Args:
            mode: "in-memory", "persistent", "http"

        Returns:

        """
        name = mode.lower()

        if name not in ChromaFactory.ALLOWED_MODES:
            raise ValueError(
                f"{name} is not an allowed mode, use one of {ChromaFactory.ALLOWED_MODES}"
            )

        self.mode = name

    def set_path(self, path: str):
        """
        Setter for the path parameter, for persistent mode
        Args:
            path: where to persist the database

        Returns:

        """
        self.path = path

    def set_host(self, host: str):
        """
        Setter for the host parameter, for client mode
        Args:
            host: host name or IP

        Returns:

        """
        self.host = host

    def set_port(self, port: int):
        """
        Setter for the port parameter, for client mode
        Args:
            port: port number

        Returns:

        """
        self.port = port

    def _get_chroma_client(self) -> API:
        """
        Helper method to get the chromadb client according to the given mode
        Returns:

        """
        if self.mode == "in-memory":
            return chromadb.Client()
        elif self.mode == "persistent":
            if not self.path:
                raise ValueError("please provide a path for chromadb to store its data")

            return chromadb.PersistentClient(self.path)

        elif self.mode == "http":
            return chromadb.HttpClient(self.host, self.port)

    def build(self, context: "BaseContext") -> VectorStore:
        """
        Construct a chromadb based vector store

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a chroma vector store object
        """
        client = self._get_chroma_client()

        context.console.verbose_print(
            f"Getting chroma vector store using {self.mode} client"
        )

        return Chroma(
            embedding_function=context.embeddings, client=client, **self.arguments
        )
