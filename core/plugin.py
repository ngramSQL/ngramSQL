import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Any, List


class BasePlugin(ABC):

    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.__config = kwargs

    @abstractmethod
    def init_gui(self):
        pass

    def get_id(self) -> str:
        return self.__class__.__name__


class PluginManager:

    def __init__(self, spark, plugin_paths: List[str]):
        self.__spark = spark
        self.__plugin_paths = plugin_paths

        self.__plugins = []
        self.__corpora = {}
        self.__data_frames = {}
        self.__data_frame_types = defaultdict(list)

        self.__udfs = {}

    def init_plugins(self):
        module_infos = pkgutil.iter_modules(self.__plugin_paths, 'plugins.')

        for finder, name, ispkg in module_infos:
            plugin_module = importlib.import_module(name)
            print(inspect.getmembers(plugin_module, lambda o: inspect.isclass(o)))

            plugin_class_names = inspect.getmembers(plugin_module, lambda o: inspect.isclass(o) and issubclass(o, BasePlugin))
            print(plugin_class_names)

            for plugin_class_name in plugin_class_names:
                plugin_class = getattr(plugin_module, plugin_class_name[0])
                self.__plugins.append(plugin_class())

        for plugin in self.__plugins:
            print(f'\t{plugin}')

        print(self.__plugins)

    def init_plugin_gui(self):
        for plugin in self.__plugins:
            plugin.init_gui()

    def register_corpus(self, dataframe, gin, tc, plugin, name):
        self.register_dataframe(dataframe, plugin, name, ['corpus'])
        self.register_dataframe(gin, plugin, name + '_gin', ['corpus_gin'])
        self.register_dataframe(tc, plugin, name + '_tc', ['corpus_tc'])
        self.__corpora.update({name: (dataframe, gin, tc)})

    def create_dataframe(self, data, schema, plugin, name, types, columns=None):
        new_df = self.__spark.createDataFrame(data, schema, columns)

        self.register_dataframe(new_df, plugin, name, types)

        return new_df

    def register_dataframe(self, dataframe, plugin, name, types=None):
        self.__data_frames.update({(plugin, name): dataframe})

        if types:
            for type in types:
                self.__data_frame_types[type].append((name, dataframe))

        dataframe.createOrReplaceTempView(name)

    def get_dataframes_by_type(self, type):
        return self.__data_frame_types[type]

    def get_corpora(self):
        return self.__corpora.keys()

    def get_corpus(self, name):
        print(name)
        print(self.__corpora)

        return self.__corpora[name]

    def get_dataframe(self, plugin, name):
        return self.__data_frames[(plugin, name)]

    def register_udf(self, udf, return_type, plugin, name):
        self.__udfs.update({plugin: name})

        self.__spark.udf(udf, return_type)


