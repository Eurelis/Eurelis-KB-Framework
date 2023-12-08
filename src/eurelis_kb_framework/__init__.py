import os
from sys import exit
from typing import Optional, TYPE_CHECKING

from .base_factory import BaseFactory, T
from .langchain_wrapper import LangchainWrapper

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class LangchainWrapperFactory(BaseFactory[LangchainWrapper]):
    def __init__(self):
        self.verbose = False
        self.config_path = None
        self.base_dir = None

    def set_config_path(self, path: str):
        """
        Setter for the configuration path
        Args:
            path: configuration path

        Returns:

        """
        self.config_path = path

    def set_verbose(self, verbose: bool):
        """
        Setter for the verbose parameter
        Args:
            verbose: to enable verbose parameter

        Returns:

        """
        self.verbose = verbose

    def set_base_dir(self, base_dir: str):
        """
        Setter for the base directory
        Args:
            base_dir: base directory

        Returns:

        """
        self.base_dir = base_dir

    @staticmethod
    def get_config_final_path(output, dirname, path: Optional[str]) -> str:
        """
        Helper method to determine configuration file path
        Args:
            output: object to print to the console
            dirname: base directory to look the configuration from
            path: optional, if not set we will look for the configuration from dirname + config/knowledge_base.json,
                can be absolute

        Returns:
            str: the absolute path of the configuration file

        """
        dirname = dirname if dirname else os.getcwd()
        if not dirname:
            output.print(
                f"Base directory not provided, please use set_base_dir(...) method"
            )
            exit(-1)

        final_path = os.path.join(dirname, "config/knowledge_base.json")

        if path:
            work_path = path if os.path.isabs(path) else os.path.join(dirname, path)
            if os.path.isdir(work_path):
                final_path = os.path.join(work_path, "knowledge_base.json")
            else:
                final_path = work_path

        if not os.path.exists(final_path) or not os.path.isfile(final_path):
            output.print(f"Unable to find a config file at '{final_path}'")
            exit(-1)

        config_folder = os.path.dirname(final_path)
        env_file = os.path.join(config_folder, ".env")
        if os.path.exists(env_file) and os.path.isfile(env_file):
            from dotenv import load_dotenv

            output.verbose_print(f"Loading environment variables from '{env_file}'")
            load_dotenv(env_file)
        else:
            output.verbose_print(
                f"No environment variable file found at '{config_folder}'"
            )

        return final_path

    def build(self, context: "BaseContext") -> LangchainWrapper:
        """
        Method to build the langchain wrapper
        Args:
            context: context object, probably None in this case

        Returns:
            an initialized LanchainWrappper instance

        """

        from eurelis_kb_framework.output import OutputFactory
        from eurelis_kb_framework.langchain_wrapper import LangchainWrapper

        output_factory = OutputFactory()
        output_factory.set_verbose(self.verbose)
        output = output_factory.build(context)

        final_path = LangchainWrapperFactory.get_config_final_path(
            output, self.base_dir, self.config_path
        )

        output.print(f"Preparing langchain wrapper")
        wrapper = LangchainWrapper()
        wrapper.set_output(output)

        output.status(
            f"Reading configuration from {str(final_path)}",
            lambda: wrapper.load_config(final_path),
        )

        return wrapper
