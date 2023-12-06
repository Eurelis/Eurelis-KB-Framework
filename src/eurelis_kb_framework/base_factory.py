from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar, Collection, TYPE_CHECKING, cast

from eurelis_kb_framework.types import PARAMS, JSON
from eurelis_kb_framework.utils import parse_param_value

T = TypeVar("T")

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class DefaultFactories(Enum):
    DOCUMENT_LOADER = ("eurelis_kb_framework.document_loaders", "GenericLoaderFactory")
    SPLITTER = ("eurelis_kb_framework.text_splitter", "GenericTextSplitterFactory")
    TRANSFORMER = (
        "eurelis_kb_framework.document_transformers",
        "GenericDocumentTransformersFactory",
    )
    VECTORSTORE = ("eurelis_kb_framework.vectorstores", "GenericVectorStoreFactory")
    LLM = ("eurelis_kb_framework.llms", "GenericLLMFactory")
    CHAIN = ("eurelis_kb_framework.chains", "GenericChainsFactory")
    EMBEDDINGS = ("eurelis_kb_framework.embeddings", "GenericEmbeddingsFactory")
    MEMORY = ("eurelis_kb_framework.memory", "GenericMemoryFactory")
    ACRONYMS = ("eurelis_kb_framework.acronyms", "AcronymsTextTransformerFactory")
    RETRIEVER = ("eurelis_kb_framework.retrievers", "GenericRetrieverFactory")


class BaseFactory(ABC, Generic[T]):
    """Interface for factories."""

    def set_params(self, params: PARAMS):
        """

        Args:
            params: json based dictionary with parameters as key-value pairs

        Returns:

        """

        for key, raw_value in params.items():  # we iterate on params
            value = parse_param_value(raw_value)  # to resolve environment variables

            function_name = f"set_{key}"  # construct the expected setter name

            if hasattr(self, function_name):  # ensure the setter exists
                try:
                    func = getattr(self, function_name)
                    func(value)  # call the setter
                except AttributeError:
                    print(f"{function_name} not found")
            else:  # if the setter was not found, we use the default unknown param handler
                self.handle_unknown_param(key, value)

    def handle_unknown_param(self, key: str, value: JSON):
        """
        Handler for unknown parameters, default implementation does nothing
        Args:
            key: name of a parameter
            value: value of a parameter

        Returns:

        """
        pass

    @abstractmethod
    def build(self, context: "BaseContext") -> T:
        """
        Methods to build something

        Args:
            context: context object, usually the current instance of langchain_wrapper

        Returns:
            something

        """


class ParamsDictFactory(BaseFactory, ABC, Generic[T]):
    """
    Base factory keeping in a dict unknown params
    """

    def __init__(self):
        """
        Constructor
        """
        self.params = dict()

    def handle_unknown_param(self, key: str, value: JSON):
        """
        Handler for unknown parameters, store them in the params dict
        Args:
            key: name of a parameter
            value: value of a parameter

        Returns:

        """
        self.params[key] = value

    def extract_params(self, keys: Collection[str]) -> PARAMS:
        """
        Helper method to extract parameters from a collection of keys
        Args:
            keys: collection of keys

        Returns:
            extracted sub dict of parameters
        """
        return {key: self.params[key] for key in self.params.keys() & keys}

    def missing_params(self, keys: Collection[str]) -> Collection[str]:
        """
        Helper method to check if all required parameters are present
        Args:
            keys: the keys to check if present

        Returns:
            Collection[str]: collection of keys not present in the parameters attribute

        """
        return set(keys).difference(self.params.keys())

    def get_optional_params(self) -> PARAMS:
        """
        Helper method to directly get optional parameters from class OPTIONAL_PARAMS attribute
        Returns:
            params (dict)
        """
        if hasattr(self.__class__, "OPTIONAL_PARAMS"):
            return self.extract_params(self.__class__.OPTIONAL_PARAMS)

        raise RuntimeError(
            f"Factory {self.__class__.__name__} does not provide OPTIONAL_PARAMS value"
        )


class ProviderFactory(ParamsDictFactory, Generic[T]):
    """
    Base factory used to delegate work to another factory given a 'provider' parameter
    """

    ALLOWED_PROVIDERS = dict()

    def _get_provider(self, context) -> BaseFactory[T]:
        """
        Helper method, will instantiate the provider factory used under the hood

        Args:
            context (BaseContext): context object, usually the current instance of langchain_wrapper

        Returns:
            BaseFactory: the factory associated with a given provider name

        """
        provider = self.params.get("provider")
        if not provider or provider not in self.__class__.ALLOWED_PROVIDERS:
            allowed_list = (
                "'" + "', '".join(self.__class__.ALLOWED_PROVIDERS.keys()) + "'"
            )
            raise ValueError(
                f"Expected one of the following provider {allowed_list} got '{provider}'"
            )

        provider_class = self.__class__.ALLOWED_PROVIDERS.get(provider)

        import inspect

        if inspect.isclass(
            provider_class
        ):  # the associated value in the dictionary is already a class
            return provider_class()  # we instantiate it

        elif isinstance(
            provider_class, str
        ):  # the associated value in the dictionary is a string
            return cast(
                BaseFactory[T], context.loader.instantiate_class("", provider_class)
            )  # we use the class loader to instantiate it, no default module name

    def build(self, context: "BaseContext") -> T:
        """
        Methods to build something

        Args:
            context (BaseContext): context object, usually the current instance of langchain_wrapper

        Returns:
            instance: object build by the under the hood provider factory

        """
        factory = self._get_provider(context)
        factory.set_params(self.params)

        return factory.build(context)
