from typing import Iterable, TYPE_CHECKING

from langchain.document_loaders.base import BaseLoader

from eurelis_kb_framework.base_factory import ParamsDictFactory
from eurelis_kb_framework.document_loaders.list.list_loader import ListLoader
from eurelis_kb_framework.types import FACTORY

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class ListLoaderFactory(ParamsDictFactory[BaseLoader]):
    """
    Helper loader factory, will allow to call a loader based
    on a single target parameter using a target list
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.targets = []
        self.loader = None
        self.loader_target_name = None

    def set_targets(self, targets: Iterable[str]):
        """
        Setter for the targets list parameters
        Args:
            targets: list of targets

        Returns:

        """
        self.targets = targets

    def set_loader(self, loader: FACTORY):
        """
        Setter for the loader factory to use under the hood
        Args:
            loader: data for the loader factory

        Returns:

        """
        self.loader = loader

    def set_target_variable(self, varname: str):
        """
        Setter for the loader parameter name corresponding to a target
        Args:
            varname: loader parameter name

        Returns:

        """
        self.loader_target_name = varname

    def _ensure_required_parameters(self):
        """
        Helper method to ensure all required parameters are given
        Returns:

        """
        missing_parameters = []
        if self.targets is None:
            missing_parameters.append("targets")
        if self.loader is None:
            missing_parameters.append("loader")
        if self.loader_target_name is None:
            missing_parameters.append("loader_target_name")

        if missing_parameters:
            raise ValueError(
                f"List document loader is missing following required parameters: {str(missing_parameters)}"
            )

    def build(self, context: "BaseContext") -> BaseLoader:
        """
        Construct the List loader
        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a list loader instance
        """
        self._ensure_required_parameters()

        return ListLoader(
            self.targets, self.loader, self.loader_target_name, self.params, context
        )
