from __future__ import annotations

import json
import os.path
from abc import ABC
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence, Union, Iterator, cast, List, Iterable, Tuple

import numpy as np
from bson import ObjectId
from langchain.chains.base import Chain
from langchain.indexes._api import _get_source_id_assigner
from langchain.llms.base import BaseLLM
from langchain.schema import Document
from langchain.schema.vectorstore import VectorStore

from eurelis_kb_framework.acronyms import AcronymsTextTransformer
from eurelis_kb_framework.acronyms.acronyms_chain_wrapper import AcronymsChainWrapper
from eurelis_kb_framework.base_factory import DefaultFactories
from eurelis_kb_framework.class_loader import ClassLoader
from eurelis_kb_framework.dataset import DatasetFactory
from eurelis_kb_framework.dataset.dataset import Dataset
from eurelis_kb_framework.types import FACTORY, EMBEDDING, DOCUMENT_MEAN_EMBEDDING
from eurelis_kb_framework.utils import parse_param_value, batched


class MetadataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class BaseContext(ABC):
    """
    Base context class
    """

    def __init__(self, class_loader: ClassLoader, console=None):
        self.loader = class_loader
        self.console = console
        self.embeddings = None
        self.vector_store = None
        self.is_verbose = False

    def copy_context(self) -> BaseContext:
        new_context = BaseContext(self.loader)
        new_context.console = self.console
        new_context.embeddings = self.embeddings
        new_context.vector_store = self.vector_store
        new_context.is_verbose = self.is_verbose

        return new_context


class LangchainWrapper(BaseContext):
    """
    Langchain wrapper, main class of the project
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__(ClassLoader())
        self._datasets: Optional[OrderedDict] = None
        self._datasets_data: Optional[Union[List[dict], dict]] = None
        self.index_fn = None
        self.project = None
        self.record_manager_db_url = None
        self.llm = None
        self.llm_factory = None
        self.chain_factory = None
        self.vector_store_data = None
        self.is_initialized = False
        self._acronyms_data = None
        self._acronyms = None

    def ensure_initialized(self):
        """
        Method to ensure the wrapper is initialized

        Raise:
            ValueError if wrapper is not initialized
        """
        if not self.is_initialized:
            raise ValueError("Langchain wrapper is not initialized")

    def set_output(self, console):
        """
        Setter for console
        Args:
            console: output object to print on the console

        Returns:

        """
        self.console = console

    def set_verbose(self, verbose):
        """
        Setter for verbose parameter
        Args:
            verbose: default to False

        Returns:

        """
        self.is_verbose = verbose

    def load_config(self, path):
        """
        Load the configuration from a json file
        Args:
            path: path of the json configuration file

        Returns:

        """

        with open(path) as config_file:
            try:
                config = json.load(config_file)
            except json.decoder.JSONDecodeError as e:
                self.console.critical_print(f"Error parsing config file: {e}")
                return

            self._parse_embeddings(config.get("embeddings"))
            self._parse_vector_store(config.get("vectorstore"))

            self._datasets_data = config.get("dataset", [])
            self._acronyms_data = config.get("acronyms", None)

            self.llm_factory = config.get("llm")
            self.chain_factory = config.get("chain", {})

            self.project = parse_param_value(config.get("project", "knowledge_base"))
            self.record_manager_db_url = parse_param_value(
                config.get("record_manager", "sqlite:///record_manager_cache.sql")
            )

            sqlite_prefix = "sqlite:///"

            if self.record_manager_db_url.startswith(sqlite_prefix):
                sqlite_length = len(sqlite_prefix)
                path = self.record_manager_db_url[sqlite_length:]
                path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
                file_folder = Path(os.path.dirname(path))
                os.makedirs(file_folder, exist_ok=True)

            self.is_initialized = True

    @property
    def datasets(self):
        if not self._datasets:
            self._datasets = self.console.status(
                f"Parsing datasets",
                lambda: DatasetFactory.build_instances(self, self._datasets_data),
            )

        return self._datasets

    @property
    def acronyms(self) -> Optional[AcronymsTextTransformer]:
        if not self._acronyms_data:
            return None

        if not self._acronyms:
            self._acronyms = LangchainWrapper.get_instance_from_factory(
                self, DefaultFactories.ACRONYMS, self._acronyms_data, mandatory=False
            )

        return self._acronyms

    def _parse_embeddings(self, embeddings: FACTORY):
        """
        Process the embeddings configuration

        Args:
            embeddings: string or dictionary for the embeddings factory

        Returns:

        """
        self.console.verbose_print(f"Reading embeddings from configuration file")

        self.embeddings = LangchainWrapper.get_instance_from_factory(
            self, DefaultFactories.EMBEDDINGS, embeddings
        )

    def _parse_vector_store(self, vector_store: FACTORY):
        """
        Process the vector store configuration

        Args:
            vector_store: string or dictionary for the vector store factory

        Returns:

        """
        self.console.verbose_print(f"Reading vectorstore from configuration file")

        self.vector_store_data = vector_store

        self.vector_store = LangchainWrapper.get_instance_from_factory(
            self, DefaultFactories.VECTORSTORE, vector_store, True
        )

    def _list_datasets(self, dataset_id: Optional[str] = None) -> Iterable[Dataset]:
        """
        Getter for the dataset objects
        Args:
            dataset_id: optional, if given we will return only the named dataset

        Returns:
            list of dataset
        """
        if not dataset_id:
            return self.datasets.values()

        dataset_list = []
        dataset = self.datasets.get(dataset_id)
        if dataset:
            dataset_list.append(dataset)

        return dataset_list

    def index_documents(self, dataset_id: Optional[str] = None):
        """
        Method to index documents to the vector store
        Args:
            dataset_id: optional, if given we will work only with the named dataset

        Returns:

        """
        self.ensure_initialized()

        dataset_index_results = OrderedDict()
        from langchain.indexes import SQLRecordManager, index

        # TODO: add lockfile

        for dataset in self._list_datasets(dataset_id):
            if not dataset.index:
                self.console.print(f"Skipping dataset '{dataset.id}'")
                continue

            if dataset.index == "cache":
                self.write_files(dataset.id)
                continue

            namespace = f"{self.project}/{dataset.name}"
            record_manager = SQLRecordManager(
                namespace, db_url=self.record_manager_db_url
            )

            record_manager.create_schema()

            def index_dataset():
                dataset_documents = dataset.lazy_load()

                def with_namespace(documents: Iterator[Document]) -> Iterator[Document]:
                    for document in documents:
                        document.metadata[
                            "namespace"
                        ] = f"{self.project}/{dataset.name}"
                        yield document

                if type(dataset.vector_store).delete == VectorStore.delete:
                    if dataset.cleanup is not None:
                        raise ValueError(
                            f"unsupported {dataset.cleanup} cleanup method, this vector store only accept None"
                        )

                    num_added = 0

                    for docs in batched(with_namespace(dataset_documents), 100):
                        num_added += len(docs)
                        dataset.vector_store.add_documents(docs)

                    return {
                        "cleanup": "None",
                        "num_added": num_added,
                        "num_updated": "-",
                        "num_skipped": "-",
                        "num_deleted": "-",
                    }

                return index(
                    with_namespace(dataset_documents),
                    record_manager,
                    dataset.vector_store,
                    cleanup=dataset.cleanup,
                    source_id_key=dataset.source_id_key,
                )

            return_value = self.console.status(
                f"Indexing '{dataset.id}' dataset using '{dataset.cleanup}' cleanup method.",
                index_dataset,
            )
            return_value["cleanup"] = str(dataset.cleanup)
            return_value["source_id_key"] = dataset.source_id_key
            dataset_index_results[dataset.id] = return_value

        self.console.print_table(
            dataset_index_results.items(),
            [
                "Dataset",
                "Cleanup",
                "Doc added",
                "Doc updated",
                "Doc skipped",
                "Doc deleted",
            ],
            lambda _, keyval: (
                keyval[0],
                str(keyval[1]["cleanup"]),
                str(keyval[1]["num_added"]),
                str(keyval[1]["num_updated"]),
                str(keyval[1]["num_skipped"]),
                str(keyval[1]["num_deleted"]),
            ),
            title="Dataset Indexing",
            show_lines=True,
        )

    def print_metadata(self, dataset_id: Optional[str] = None):
        """
        Method to display the metadata of the first document for a dataset (for dev reason)
        Args:
            dataset_id: optional, if given we will work only with the named dataset

        Returns:

        """
        self.ensure_initialized()

        for dataset in self._list_datasets(dataset_id):
            first_doc = dataset.get_first_doc()

            if not first_doc:
                continue

            metadata = first_doc.metadata

            self.console.print_table(
                metadata.items(),
                ["Key", "Type", "Value"],
                lambda index, keyval: (
                    keyval[0],
                    keyval[1].__class__.__name__,
                    str(keyval[1]),
                ),
                title=f"Metadata for {dataset.id}",
                show_lines=True,
            )

    def write_files(self, dataset_id: Optional[str] = None):
        """
        Method to write dataset extracted documents to files
        Args:
            dataset_id: optional, if given we will work only with the named dataset

        Returns:

        """
        self.ensure_initialized()

        for dataset in self._list_datasets(dataset_id):
            if not dataset.output_folder:
                self.console.print(f"No output configured for dataset '{dataset.id}'")
                continue
            # TODO: create a table with number of wrote files?
            dataset.write_files(self.console)

    def _build_search_args(
        self, k: int = 4, search_filter: Optional[dict[str, str]] = None
    ) -> dict:
        search_args = {"k": k}

        import inspect

        argspect = inspect.getfullargspec(self.vector_store.similarity_search)
        if search_filter:
            is_filter = "filter" in argspect.args
            is_where = "where" in argspect.args
            if not is_filter and is_where:
                raise RuntimeError(
                    f"Used vector store does allow to support 'filter' arguments"
                )
            search_args["filter" if is_filter else "where"] = search_filter

        return search_args

    def metadata_search_documents(
        self, k: int = 10, search_filter: Optional[dict[str, str]] = None
    ) -> List[Document]:
        """
        Method to fetch k document using filters vector store
        Args:
            k: max number of documents to return
            search_filter: filter

        Returns:
            list of documents
        """

        self.ensure_initialized()

        search_args = self._build_search_args(k, search_filter)

        return self.vector_store.similarity_search(query="", **search_args)

    def mean_embedding_from_metadata_search_documents(
        self,
        k: int = 10,
        search_filter: Optional[dict[str, str]] = None,
        mean_embedding_method: DOCUMENT_MEAN_EMBEDDING = "default",
    ) -> Optional[EMBEDDING]:
        """
        Method to fetch k document using filters vector store and compute a mean embedding
        Args:
            k: max number of documents to return
            search_filter: filter
            mean_embedding_method: you can pass a method taking an Embeddings object and a sequence of documents
                and returning a vector

        Returns:
            embedding, list of float
        """
        # no need to call ensure initialized as it is performed on metadata search documents
        documents = self.metadata_search_documents(k, search_filter)

        if not documents:
            return None

        # process the embeddings

        if callable(mean_embedding_method):
            return mean_embedding_method(self.embeddings, documents)

        # use default mean embedding

        doc_contents = list(map(lambda doc: doc.page_content, documents))
        embeddings = self.embeddings.embed_documents(doc_contents)

        return np.mean(np.array(embeddings), axis=0).tolist()

    def get_embedding_associated_documents(
        self,
        *sources: str,
        source_field: str = "source",
        k: int = 4,
        expected_docs_by_source: int = 1,
        single_doc_by_source: bool = True,
        source_mean_embedding_method: DOCUMENT_MEAN_EMBEDDING = "default",
    ) -> List[Tuple[Document, float]]:
        """
        Method to get suggestions

        Args:
            *sources(str): list of sources
            source_field(str): the metadata field to use to filter sources for
            k(int): the expected number of documents to return
            expected_docs_by_source(int): the expected number of documents for a given source
            single_doc_by_source(bool): default to True, only one document for a given source in the result list
            source_mean_embedding_method: to specify a callable to compute mean embedding for a document sequence

        Returns:
            list of tuples with the document and relevance score
        """

        # get the mean embedding for the sources
        embeddings = []
        for source in sources:
            embedding = self.mean_embedding_from_metadata_search_documents(
                k=expected_docs_by_source,
                search_filter={source_field: source},
                mean_embedding_method=source_mean_embedding_method,
            )

            if embedding:
                embeddings.append(embedding)

        if not embeddings:
            return []

        mean_embedding = np.mean(np.array(embeddings), axis=0).tolist()

        # get the docs nearest to the mean embedding
        k_search_documents = (
            k + len(sources)
        ) * expected_docs_by_source  # we ask for more documents in case we need to filter out source docments
        search_documents = self.vector_search_documents(
            mean_embedding, k_search_documents, include_relevance=True
        )

        # we remove from documents those corresponding to the given sources
        documents = filter(
            lambda item: item[0].metadata.get(source_field) not in sources,
            search_documents,
        )

        if single_doc_by_source:
            new_docs = []
            source_set = set()

            # ensure to have only the first document for a given source value
            for doc in documents:
                source = doc[0].metadata.get(source_field)
                if source not in source_set:
                    new_docs.append(doc)
                    source_set.add(source)

            documents = new_docs

        # we truncate the return list
        return list(documents)[:k]

    def vector_search_documents(
        self,
        vector: List[float],
        k: int = 4,
        search_filter: Optional[dict[str, str]] = None,
        include_relevance: Optional[bool] = False,
    ):
        """
        Method to execute a similarity search on the vector store
        Args:
            vector: the query to look documents for
            k: max number of documents to return
            search_filter: filter
            include_relevance: boolean, default

        Returns:
            list of documents
        """

        self.ensure_initialized()

        search_args = self._build_search_args(k, search_filter)

        search_method = (
            self.vector_store.similarity_search
            if not include_relevance
            else self.vector_store.similarity_search_by_vector_with_relevance_scores
        )

        documents = search_method(vector, **search_args)

        return documents

    def search_documents(
        self,
        query: str,
        k: int = 4,
        search_filter: Optional[dict[str, str]] = None,
        for_print: bool = False,
        include_relevance: bool = False,
    ):
        """
        Method to execute a similarity search on the vector store
        Args:
            query (str): the query to look documents for
            k (int): max number of documents to return
            search_filter: filter
            for_print (bool): default to False, if True method will always print result in the console
            include_relevance (bool): should we include relevance in results

        Returns:
            list of documents
        """

        self.ensure_initialized()

        search_args = self._build_search_args(k, search_filter)

        search_method = (
            self.vector_store.similarity_search
            if not include_relevance
            else self.vector_store.similarity_search_with_relevance_scores
        )

        documents = self.console.status(
            "Performing similarity search",
            lambda: search_method(query=query, **search_args),
        )

        console_print_table = (
            self.console.print_table if for_print else self.console.verbose_print_table
        )

        console_print_table(
            documents,
            ["Index", "Content", "Metadata"],
            lambda index, document: (
                str(index),
                document.page_content,
                json.dumps(document.metadata, cls=MetadataEncoder),
            ),
            title=query,
        )

        return documents

    def list_datasets(self):
        """
        Method to print the list of datasets
        Returns:

        """

        self.ensure_initialized()

        datasets = self._list_datasets()
        Dataset.print_datasets(self.console, datasets, verbose_only=False)

    def _delete_from_namespace(
        self, namespace: str, documents: List[Document], dataset_id: Optional[str]
    ) -> int:
        """
        Delete documents inside a namespace
        Args:
            namespace: namespace
            documents: list of documents
            dataset_id: optional dataset id

        Returns:
            number of deleted documents
        """

        self.ensure_initialized()

        from langchain.indexes import SQLRecordManager

        # TODO: add lockfile

        # TODO: get dataset from document schema

        num_deleted = 0

        dataset = next(
            (
                dataset
                for dataset in self._list_datasets(dataset_id)
                if dataset.index
                and dataset.index != "cache"
                and f"{self.project}/{dataset.name}" == namespace
            ),
            None,
        )

        if not dataset:
            return 0

        source_id_assigner = _get_source_id_assigner(dataset.source_id_key)

        source_ids: Sequence[Optional[str]] = [
            source_id_assigner(doc) for doc in documents
        ]

        namespace = f"{self.project}/{dataset.name}"
        record_manager = SQLRecordManager(namespace, db_url=self.record_manager_db_url)

        record_manager.create_schema()

        _source_ids = cast(Sequence[str], source_ids)

        uids_to_delete = record_manager.list_keys(group_ids=_source_ids)

        if uids_to_delete:
            # Then delete from vector store.
            self.vector_store.delete(uids_to_delete)
            # First delete from record store.
            record_manager.delete_keys(uids_to_delete)
            num_deleted += len(uids_to_delete)

        return num_deleted

    def delete(self, search_filter: dict[str, str], dataset_id: Optional[str]) -> int:
        """
        Method to delete documents using a search query
        Args:
            search_filter: filters for the search query
            dataset_id: optional dataset id

        Returns:

        """
        self.ensure_initialized()

        if not filter:
            raise ValueError(f"Missing delete filter value")

        def delete_work():
            total_num_deleted = 0
            should_search = True

            while should_search:
                num_deleted = 0
                documents = self.metadata_search_documents(
                    k=10,
                    search_filter=search_filter,
                )

                namespace_set = set()

                for doc in documents:
                    namespace_set.add(doc.metadata.get("namespace"))

                for namespace in namespace_set:
                    num_deleted += self._delete_from_namespace(
                        namespace, documents, dataset_id
                    )

                total_num_deleted += num_deleted
                should_search = bool(num_deleted)

            return total_num_deleted

        final_num_deleted = self.console.status("Processing delete query", delete_work)

        self.console.print(f"{final_num_deleted} chunk(s) deleted from database")

        return final_num_deleted

    def clear_datasets(self, dataset_id: Optional[str] = None):
        """
        Method to clear documents in datasets
        Args:
            dataset_id: optional, if given we will work only with the named dataset

        Returns:

        """
        self.ensure_initialized()

        dataset_index_results = OrderedDict()
        from langchain.indexes import SQLRecordManager, index

        # TODO: add lockfile

        for dataset in self._list_datasets(dataset_id):
            if not dataset.index:
                self.console.print(f"Skipping dataset '{dataset.id}'")
                continue

            namespace = f"{self.project}/{dataset.name}"
            record_manager = SQLRecordManager(
                namespace, db_url=self.record_manager_db_url
            )

            record_manager.create_schema()

            def clear_dataset():
                return index(
                    [],
                    record_manager,
                    self.vector_store,
                    cleanup="full",
                    source_id_key=dataset.source_id_key,
                )

            return_value = self.console.status(
                f"Clearing '{dataset.id}' dataset", clear_dataset
            )
            return_value["cleanup"] = str(dataset.cleanup)
            return_value["source_id_key"] = dataset.source_id_key
            dataset_index_results[dataset.id] = return_value

        self.console.print_table(
            dataset_index_results.items(),
            [
                "Dataset",
                "Cleanup",
                "Doc added",
                "Doc updated",
                "Doc skipped",
                "Doc deleted",
            ],
            lambda _index, keyval: (
                keyval[0],
                str(keyval[1]["cleanup"]),
                str(keyval[1]["num_added"]),
                str(keyval[1]["num_updated"]),
                str(keyval[1]["num_skipped"]),
                str(keyval[1]["num_deleted"]),
            ),
            title="Dataset Clearing",
            show_lines=True,
        )

    def lazy_get_llm(self) -> BaseLLM:
        """
        Helper method to get the LLM instance when needed

        Returns:
            BaseLLM: llm instance
        """
        self.ensure_initialized()

        if not self.llm and self.llm_factory:
            self.llm = LangchainWrapper.get_instance_from_factory(
                self, DefaultFactories.LLM, self.llm_factory, mandatory=True
            )

        return self.llm

    def get_chain(self, **kwargs) -> Chain:
        """
        Helper method to get a langchain chain

        Args:
            kwargs: additional parameters to use on chain creation, can override default values from the json config
        """
        self.ensure_initialized()

        chain_args = {**self.chain_factory, **kwargs}

        chain = LangchainWrapper.get_instance_from_factory(
            self, DefaultFactories.CHAIN, chain_args, mandatory=True
        )

        if self.acronyms:
            # instantiate and return a wrapping chain to handle acronyms
            return AcronymsChainWrapper(chain, self.acronyms)

        return chain

    @staticmethod
    def get_instance_from_factory(
        context,
        default: DefaultFactories,
        data: Optional[FACTORY],
        mandatory: bool = False,
    ):
        """
        Helper method to
        Args:
            context: the context object to build the instance, usually the langchain wrapper itself
            default: default factories enum value, contains default module and class name
            data: factory data to provide, either a string or a dictionary
            mandatory (bool): if False no instance is returned if no data is provided,
                else we construct the object using only default factory values

        Returns:

        """
        if not mandatory and not data:
            return None

        # instantiate the factory
        class_loader = context.loader
        default_values = default.value
        factory = class_loader.instantiate_factory(
            default_values[0], default_values[1], data if data else {}
        )

        # use the factory to build the object
        return factory.build(context)
