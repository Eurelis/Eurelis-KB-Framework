from __future__ import annotations

import json
import os.path
from abc import ABC
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence, Union, Iterator, cast, List

from langchain.chains.base import Chain
from langchain.indexes._api import _get_source_id_assigner
from langchain.llms import BaseLLM
from langchain.schema import Document

from eurelis_kb_framework.base_factory import DefaultFactories, FACTORY
from eurelis_kb_framework.class_loader import ClassLoader
from eurelis_kb_framework.dataset import DatasetFactory
from eurelis_kb_framework.dataset.dataset import Dataset


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
        self.datasets = OrderedDict()
        self.index_fn = None
        self.project = None
        self.record_manager_db_url = None
        self.llm = None
        self.llm_factory = None
        self.chain_factory = None
        self.vector_store_data = None

    def set_console(self, console):
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
            config = json.load(config_file)

            self._parse_embeddings(config.get("embeddings"))
            self._parse_vector_store(config.get("vectorstore"))

            self._parse_dataset(config.get("dataset", []))

            self.llm_factory = config.get("llm")
            self.chain_factory = config.get("chain", {})

            self.project = config.get("project", "knowledge_base")
            self.record_manager_db_url = config.get(
                "record_manager", "sqlite:///record_manager_cache.sql"
            )

            sqlite_prefix = "sqlite:///"

            if "record_manager" in config and config["record_manager"].startswith(
                sqlite_prefix
            ):
                sqlite_length = len(sqlite_prefix)
                path = config["record_manager"][sqlite_length:]
                path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
                file_folder = Path(os.path.dirname(path))
                os.makedirs(file_folder, exist_ok=True)

    def _parse_dataset(self, datasets: Union[FACTORY, iter[FACTORY]]):
        """
        Parse dataset configuration
        Args:
            datasets: either a single dataset dict or a list of dataset dict

        Returns:

        """
        if not datasets:
            return
        self.datasets = DatasetFactory.build_instances(self, datasets)

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

    def _list_datasets(self, dataset_id: Optional[str] = None) -> Sequence[Dataset]:
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
        for dataset in self._list_datasets(dataset_id):
            if not dataset.output_folder:
                self.console.print(f"No output configured for dataset '{dataset.id}'")
                continue
            # TODO: create a table with number of wrote files?
            dataset.write_files(self.console)

    def vector_search_documents(
        self,
        vector: List[float],
        k: int = 4,
        filter: Optional[dict[str, str]] = None,
        include_relevance: Optional[bool] = False,
    ):
        """
        Method to execute a similarity search on the vector store
        Args:
            vector: the query to look documents for
            filter: filter
            include_relevance: boolean, default

        Returns:
            list of documents
        """

        similarity_search_args = {"k": k}

        import inspect

        argspect = inspect.getfullargspec(self.vector_store.similarity_search)
        if filter:
            is_filter = "filter" in argspect.args
            is_where = "where" in argspect.args
            if not is_filter and is_where:
                raise RuntimeError(
                    f"Used vector store does allow to support 'filter' arguments"
                )
            similarity_search_args["filter" if is_filter else "where"] = filter

        search_method = (
            self.vector_store.similarity_search
            if not include_relevance
            else self.vector_store.similarity_search_by_vector_with_relevance_scores
        )

        documents = search_method(vector, **similarity_search_args)

        return documents

    def search_documents(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, str]] = None,
        for_print: bool = False,
        for_delete: bool = False,
    ):
        """
        Method to execute a similarity search on the vector store
        Args:
            query: the query to look documents for
            filter: filter
            for_print: default to False, if True method will always print result in the console
            for_delete: default to False, if True status won't be displayed while searching

        Returns:
            list of documents
        """

        similarity_search_args = {"k": k}

        import inspect

        argspect = inspect.getfullargspec(self.vector_store.similarity_search)
        if filter:
            is_filter = "filter" in argspect.args
            is_where = "where" in argspect.args
            if not is_filter and is_where:
                raise RuntimeError(
                    f"Used vector store does allow to support 'filter' arguments"
                )
            similarity_search_args["filter" if is_filter else "where"] = filter

        if not for_delete:
            documents = self.console.status(
                "Performing similarity search",
                lambda: self.vector_store.similarity_search(
                    query=query, **similarity_search_args
                ),
            )
        else:
            documents = self.vector_store.similarity_search(
                query=query, **similarity_search_args
            )

        console_print_table = (
            self.console.print_table if for_print else self.console.verbose_print_table
        )

        if not for_delete:
            console_print_table(
                documents,
                ["Index", "Content", "Metadata"],
                lambda index, document: (
                    str(index),
                    document.page_content,
                    json.dumps(document.metadata),
                ),
                title=query,
            )

        return documents

    def list_datasets(self):
        """
        Method to print the list of datasets
        Returns:

        """
        datasets = self._list_datasets()
        Dataset.print_datasets(self.console, datasets, verbose_only=False)

    def delete_from_namespace(
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
        print("=============")
        print(uids_to_delete)

        if uids_to_delete:
            # Then delete from vector store.
            self.vector_store.delete(uids_to_delete)
            # First delete from record store.
            record_manager.delete_keys(uids_to_delete)
            num_deleted += len(uids_to_delete)

        return num_deleted

    def delete(self, filter: dict[str, str], dataset_id: Optional[str]) -> int:
        """
        Method to delete documents using a search query
        Args:
            filter: filters for the search query
            dataset_id: optional dataset id

        Returns:

        """
        if not filter:
            raise ValueError(f"Missing delete filter value")

        def delete_work():
            total_num_deleted = 0
            should_search = True

            while should_search:
                num_deleted = 0
                documents = self.search_documents(
                    "", filter, for_print=False, for_delete=True
                )

                namespace_set = set()

                for doc in documents:
                    namespace_set.add(doc.metadata.get("namespace"))

                for namespace in namespace_set:
                    num_deleted += self.delete_from_namespace(
                        namespace, documents, dataset_id
                    )

                total_num_deleted += num_deleted
                should_search = bool(num_deleted)

            return total_num_deleted

        total_num_deleted = self.console.status("Processing delete query", delete_work)

        self.console.print(f"{total_num_deleted} chunk(s) deleted from database")

        return total_num_deleted

    def clear_datasets(self, dataset_id: Optional[str] = None):
        """
        Method to clear documents in datasets
        Args:
            dataset_id: optional, if given we will work only with the named dataset

        Returns:

        """
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
            lambda index, keyval: (
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
        if not self.llm and self.llm_factory:
            self.llm = LangchainWrapper.get_instance_from_factory(
                self, DefaultFactories.LLM, self.llm_factory, mandatory=True
            )

        return self.llm

    def get_chain(self, **kwargs) -> Chain:
        chain_args = {**self.chain_factory, **kwargs}

        return LangchainWrapper.get_instance_from_factory(
            self, DefaultFactories.CHAIN, chain_args, mandatory=True
        )

    @staticmethod
    def get_instance_from_factory(
        context, default: DefaultFactories, data: FACTORY, mandatory=False
    ):
        """
        Helper method to
        Args:
            context:
            default:
            data:
            mandatory:

        Returns:

        """
        if not mandatory and not data:
            return None

        class_loader = context.loader
        default_values = default.value
        factory = class_loader.instantiate_factory(
            default_values[0], default_values[1], data if data else {}
        )

        return factory.build(context)
