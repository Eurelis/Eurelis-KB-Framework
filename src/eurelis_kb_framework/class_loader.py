from __future__ import annotations

import importlib
from types import ModuleType
from typing import Union

from eurelis_kb_framework.base_factory import BaseFactory


class ClassLoader:
    """
    Method to dynamically instantiate classes by either their module and name or fully qualified name
    """

    def __init__(self):
        self.loaded_modules = dict()

    def load_module(self, module_name: str) -> ModuleType:
        """
        Load a module from its name, cache the result

        Args:
            module_name: import path of a module to load

        Returns:
            module object
        """
        if module_name not in self.loaded_modules:
            self.loaded_modules[module_name] = importlib.import_module(module_name)

        return self.loaded_modules.get(module_name)

    def instantiate_class(self, default_module_name: str, class_data: Union[str, dict]):
        """
        Instantiate a class

        Args:
            default_module_name: default module name for the class if a fully qualified name is not given
            class_data: either a string (name or fully qualified name of the class) either a dictionary to construct the class from

        Returns:
            a class instance
        """

        if isinstance(class_data, str):
            return self._instantiate_class(default_module_name, class_data)
        elif isinstance(class_data, dict):
            return self._instantiate_class_from_dict(default_module_name, class_data)

        raise ValueError(f"invalid data found {type(class_data)} expecting string or dictionary")

    def _instantiate_class(self, default_module_name: str, qualified_class_name: str, *args, **kwargs):
        """
        Instantiate a class with its name and arguments

        Args:
            default_module_name: the default module name to use if the class name isn't qualified
            qualified_class_name: the class name, can be qualified or not
            *args: positional arguments to provide the class for instantiation
            **kwargs: keyword arguments to provide the class for instantiation

        Returns:
            a class instance
        """
        class_name_split = qualified_class_name.split(".")
        class_name = class_name_split[-1]
        module_name = ".".join(class_name_split[:-1]) if len(class_name_split) >= 2 else default_module_name

        module = self.load_module(module_name)

        class_ = getattr(module, class_name)
        if not class_:
            raise ValueError(f"module {module_name} doest not contain a {class_name} class")

        return class_(*args, **kwargs)

    def _instantiate_class_from_dict(self, default_module_name: str, class_dict: dict):
        """
        Instantiate a class from an object representation
        Args:
            default_module_name: the default module name of the class
            class_dict: object representation of the class, must contain a value for the 'class' key, can contain values for 'args' and 'kwargs' values

        Returns:
            a class instance

        """
        # extract class name and arguments
        qualified_class = class_dict.pop('class')
        args = class_dict.pop('args', list())
        kwargs = class_dict.pop('kwargs', dict())

        if not qualified_class:
            raise ValueError(f"missing class attribute in {class_dict} instance definition")

        # delegate the instantiation
        return self._instantiate_class(default_module_name, qualified_class, *args, **kwargs)

    def instantiate_factory(self, default_module_name: str, default_factory_class: str, factory_data: str | dict) -> BaseFactory:
        """
        Instantiate a factory object
        Args:
            default_module_name: the default module name for the factory class
            default_factory_class: the default class name for the factory class
            factory_data: either a string (factory class name or fully qualified factory class name) or a dictionary to construct the factory

        Returns:
            a factory object
        """
        if isinstance(factory_data, str):
            return self._instantiate_factory(default_module_name, factory_data)
        elif isinstance(factory_data, dict):
            return self._instantiate_factory_from_dict(default_module_name, default_factory_class, factory_data)

        raise ValueError(f"invalid data found {type(factory_data)} expecting string or dictionary")

    def _instantiate_factory(self, default_module_name: str, qualified_class_name: str, *args, **kwargs) -> BaseFactory:
        """
        Instantiate a factory with its default module, its name and its arguments

       Args:
           default_module_name: the default module name to use if the factory class name isn't qualified
           qualified_class_name: the factory class name, can be qualified or not
           *args: positional arguments to provide the factory class for instantiation (shouldn't be needed)
           **kwargs: keyword arguments to provide the factory class for instantiation (shouldn't be needed)

       Returns:
           a factory instance
        """
        class_name_split = qualified_class_name.split(".")
        class_name = class_name_split[-1]
        module_name = ".".join(class_name_split[:-1]) if len(class_name_split) >= 2 else default_module_name

        module = self.load_module(module_name)

        class_ = getattr(module, class_name)
        if not class_:
            raise ValueError(f"module {module_name} doest not contain a {class_name} class")

        return class_(*args, **kwargs)

    def _instantiate_factory_from_dict(self, default_module_name: str, default_factory_class: str, factory_dict: dict):
        """
        Instantiate a factory with its module, its name and an object representation

       Args:
           default_module_name: the default module name to use if the factory class name isn't qualified
           default_factory_class: the factory class name, can be qualified or not
           factory_dict: object representation of the factory class, can contain a value for the 'factory' key (to override the default factory class), can contain values for 'args' and 'kwargs' values

       Returns:
           a factory instance
        """
        qualified_class = factory_dict.pop('factory', default_factory_class)
        args = factory_dict.pop('args', list())
        kwargs = factory_dict.pop('kwargs', dict())

        if not qualified_class:
            raise ValueError(f"empty factory attribute in {factory_dict} instance definition")

        instance = self._instantiate_class(default_module_name, qualified_class, *args, **kwargs)

        instance.set_params(factory_dict)

        return instance
