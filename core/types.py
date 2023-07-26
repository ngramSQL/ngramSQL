from typing import Iterable, Any, Tuple

from pyspark.sql import DataFrame


class Corpus:

    def __init__(self, identifier: str) -> None:
        self.__identifier = identifier


class Token:

    def __init__(self, token_string: str, token_type: str) -> None:
        self.__string = token_string
        self.__type = token_type


class Ngram(Tuple[Token]):

    def __init__(self, tokens: Iterable[Any]) -> None:
        super(tokens)


class TimeSeries:
    pass


class NgramDataFrame(DataFrame):
    pass



