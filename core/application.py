from __future__ import annotations

import os
from typing import Final

import toml
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException, ParseException

from ngramsql.core.mainframe import MainFrame
from ngramsql.core.plugin import PluginManager
from ngramsql.core.util import toml_dict_to_dotted_strings


class Application:
    __app = None
    __main_frame = None
    __plugin_manager = None

    __CONFIG_PREFIX: Final[str] = 'ngramsql'

    __CONFIG_CORE_PREFIX: Final[str] = 'core'
    __CONFIG_SPARK_PREFIX: Final[str] = 'spark'

    __PLUGIN_PATH = 'plugins'

    def __init__(self) -> None:
        if Application.__app:
            raise AssertionError(f'{self.__class__} was already started!')
        Application.__app = self

        self.__tabs = ['Explore', 'SQL']

        print(os.getcwd())
        with open("pyproject.toml", "rb") as f:
            self.__config = toml.loads(f.read().decode('utf8'))[self.__CONFIG_PREFIX]

        self.__core_config = self.__config[self.__CONFIG_CORE_PREFIX]
        self.__spark_config = {self.__CONFIG_SPARK_PREFIX: self.__core_config[self.__CONFIG_SPARK_PREFIX]}

        self.__spark = self.__launch_spark()

        self.read = self.__spark.read

        Application.__plugin_manager = PluginManager(self.__spark, [self.__PLUGIN_PATH])
        Application.__plugin_manager.init_plugins()

        # Application.__main_frame = MainFrame(self)

    def __launch_spark(self) -> SparkSession:
        spark_config_strings = toml_dict_to_dotted_strings(self.__spark_config)
        conf = SparkConf()
        conf.setAll(spark_config_strings)

        return SparkSession.builder.appName('ngramSQL').config(conf=conf).getOrCreate()

    @staticmethod
    def get_or_create() -> Application:
        if not Application.__app:
            _ = Application()

        #_main_frame = MainFrame(Application.__app)
        # Application.__plugin_manager.init_plugin_gui()

        return Application.__app

    def set_main_frame(self, main_frame):
        Application.__main_frame = main_frame

        Application.__plugin_manager.init_plugin_gui()

    def loop(self):
        while True:
            query = input('> ')

            try:
                self.execute_sql(query).show()
            except AnalysisException as e:
                print(e)
            except ParseException as e:
                print(e)

    def execute_sql(self, query):
        result = self.__spark.sql(query)

        return result

    def get_specific_config(self, tl_prefix):
        if tl_prefix not in self.__config:
            return {}

        return self.__config[tl_prefix]

    def get_corpus(self, name):
        return self.__plugin_manager.get_corpus(name)

    def get_corpora(self):
        return self.__plugin_manager.get_corpora()

    def get_union_of_corpora(self):
        corpora = self.__plugin_manager.get_corpora()
        print(corpora)

        df_to_return = None

        for corpus in corpora:
            df_to_add = self.__plugin_manager.get_corpus(corpus)

            if not df_to_return:
                df_to_return = df_to_add
            else:
                df_to_return.union(df_to_add)

        return df_to_return

    def create_dataframe(self, data, schema, plugin, name, types=[], columns=None):
        return self.__plugin_manager.create_dataframe(data, schema, plugin.get_id(), name, types, columns)

    def create_corpus(self, data, schema, plugin, name):
        return self.__plugin_manager.create_corpus(data, schema, plugin.get_id(), name)

    def register_dataframe(self, dataframe, plugin, name, types):
        self.__plugin_manager.register_dataframe(dataframe, plugin.get_id(), name, types)

    def get_dataframes_by_type(self, type):
        return self.__plugin_manager.get_dataframes_by_type(type)

    def register_corpus(self, dataframe, gin, tc, plugin, name):
        self.__plugin_manager.register_corpus(dataframe, gin, tc, plugin.get_id(), name)

    def register_udf(self, udf, return_type, plugin, name):
        self.__plugin_manager.register_udf(udf, return_type, plugin.get_id(), name)

    def add_tab(self, tab_name):
        self.__tabs.append(tab_name)

    def get_tab(self, tab_name):
        if not Application.__main_frame:
            return None

        return Application.__main_frame.get_tab(tab_name)

    def get_tabs(self):
        return self.__tabs





