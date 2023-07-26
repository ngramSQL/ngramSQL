from typing import Dict, Any

from core.application import Application
from core.plugin import BasePlugin
from plugins.google_books.fileio import RawCsvDataSource


class GoogleBooksPlugin(BasePlugin):

    __CONFIG_PREFIX = 'google_books'

    __CORPUS_CONFIG_PREFIX = 'corpus'
    __TYPE_CONFIG_PREFIXES = ['csv']
    __IDENTIFIERS_CONFIG_KEY = 'identifiers'
    __ROOTS_CONFIG_KEY = 'roots'

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.__config = Application.get_or_create().get_specific_config(self.__CONFIG_PREFIX)

        self.__corpus_config = self.__config[self.__CORPUS_CONFIG_PREFIX]

        for corpus_type in self.__TYPE_CONFIG_PREFIXES:
            if corpus_type not in self.__corpus_config:
                continue

            current_config = self.__corpus_config[corpus_type]

            if corpus_type == 'csv':
                identifiers = current_config[self.__IDENTIFIERS_CONFIG_KEY]
                roots = current_config[self.__ROOTS_CONFIG_KEY]

                assert len(identifiers) == len(roots)

                for i in range(len(identifiers)):
                    corpus, gin, tc = RawCsvDataSource(roots[i], identifiers[i]).read()

                    Application.get_or_create().register_corpus(corpus, gin, tc, self, identifiers[i])

    def init_gui(self):
        pass
