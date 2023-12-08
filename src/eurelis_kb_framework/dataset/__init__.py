from collections import OrderedDict
from typing import TYPE_CHECKING

from eurelis_kb_framework.acronyms.acronyms_document_transformer import (
    AcronymsDocumentTransformer,
)
from eurelis_kb_framework.base_factory import (
    ParamsDictFactory,
    DefaultFactories,
    JSON,
)
from eurelis_kb_framework.dataset.dataset import Dataset
from eurelis_kb_framework.types import FACTORY

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class DatasetFactory(ParamsDictFactory[Dataset]):
    def __init__(self):
        super().__init__()
        self.id = None
        self.loader_factory_data = None
        self.splitter_factory_data = None
        self.transformer_factory_data = None
        self.embeddings_data = None
        self.output_folder = None
        self.output_file_varname = "id"
        self.source_id_key = "id"
        self.metadata = None

    def set_id(self, dataset_id: str):
        """
        Setter for the id parameter
        Args:
            dataset_id: id of the dataset

        Returns:

        """
        self.id = dataset_id

    def set_loader(self, loader: FACTORY):
        """
        Setter for the document loader factory data
        Args:
            loader: data for the document loader factory

        Returns:

        """
        self.loader_factory_data = loader

    def set_transformer(self, transformer: FACTORY):
        """
        Setter for the transformer factory data
        Args:
            transformer: data for the transformer factory

        Returns:

        """
        self.transformer_factory_data = transformer

    def set_splitter(self, splitter: FACTORY):
        """
        Setter for the splitter factory data
        Args:
            splitter: data for the splitter factory

        Returns:

        """
        self.splitter_factory_data = splitter

    def set_embeddings(self, embeddings: FACTORY):
        """
        Setter for the embeddings factory data
        Args:
            embeddings: data for the embeddings factory

        Returns:

        """
        self.embeddings_data = embeddings

    def set_metadata(self, metadata: FACTORY):
        """
        Setter for the metadata variable
        Args:
            metadata:

        Returns:
        """
        self.metadata = metadata

    def _handle_output(self, instance: Dataset):
        """
        Helper method to handle output related options
        Args:
            instance: dataset instance

        Returns:

        """
        output = self.params.get("output")

        output_folder = None
        output_file_varname = "id"

        if output:
            if isinstance(output, str):
                output_folder = output
            elif isinstance(output, dict):
                output_folder = output.get("folder")
                output_file_varname = output.get("varname", "id")

        instance.set_output_folder(output_folder)
        instance.set_output_file_varname(output_file_varname)

    def _handle_index(self, instance: Dataset):
        """
        Helper method to handle index related options
        Args:
            instance: dataset instance

        Returns:

        """
        index = self.params.get("index")

        instance.set_index(index)

    def build(self, context: "BaseContext") -> Dataset:
        """
        Method to build the dataset object
        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a dataset instance

        """
        loader = context.__class__.get_instance_from_factory(
            context,
            DefaultFactories.DOCUMENT_LOADER,
            self.loader_factory_data,
            mandatory=True,
        )
        splitter = context.__class__.get_instance_from_factory(
            context, DefaultFactories.SPLITTER, self.splitter_factory_data
        )
        transformer = context.__class__.get_instance_from_factory(
            context, DefaultFactories.TRANSFORMER, self.transformer_factory_data
        )

        acronyms = context.acronyms
        if acronyms:
            transformer = AcronymsDocumentTransformer(acronyms, transformer)

        instance = Dataset(self.id, loader)

        if self.embeddings_data:
            embeddings = context.__class__.get_instance_from_factory(
                context, DefaultFactories.EMBEDDINGS, self.embeddings_data
            )
            if embeddings:
                local_context = context.copy_context()
                local_context.embeddings = embeddings
                vector_store = context.__class__.get_instance_from_factory(
                    local_context,
                    DefaultFactories.VECTORSTORE,
                    context.vector_store_data,
                )
                instance.set_vector_store(vector_store)

        if not instance.vector_store:
            instance.set_vector_store(context.vector_store)

        instance.set_splitter(splitter)
        instance.set_transformer(transformer)
        instance.set_metadata(self.metadata)

        self._handle_output(instance)
        self._handle_index(instance)

        return instance

    @staticmethod
    def build_instances(context, data: JSON) -> OrderedDict[str, Dataset]:
        """
        Helper method to build datasets from configuration file
        Args:
            context: the context object, usually the current langchain wrapper instance
            data: json object or list of objects found in configuration file

        Returns:

        """
        result = OrderedDict()

        if not data:
            return result

        dataset_data_list = [data] if isinstance(data, dict) else data

        for dataset_data in dataset_data_list:
            factory = DatasetFactory()
            factory.set_params(dataset_data)

            instance = factory.build(context)

            result[instance.id] = instance

        return result
