import functools
import logging
import os.path
from operator import floordiv
from typing import Dict, Any, List, Tuple

from pyspark.sql import DataFrame, Window, functions
from pyspark.sql.functions import split, col, element_at, slice, size, regexp_extract, transform, when, explode, \
    monotonically_increasing_id, map_from_arrays, lit, udf, collect_list, row_number, ceil, array
from pyspark.sql.types import IntegerType, ArrayType, LongType

from core.application import Application
from core.types import Ngram, Corpus, TimeSeries

class RawCsvDataSource:
    __PARTS_OF_SPEECH = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'CONJ', 'PRT', 'X']
    __POS_SPLIT_REGEX = f'(.*)({"|".join(["_" + pos for pos in __PARTS_OF_SPEECH])})$'

    def __init__(self, path, identifier, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.__path = path
        self.__data_parquet_path = os.path.join(path, 'data.parquet')
        self.__gin_parquet_path = os.path.join(path, 'gin.parquet')
        self.__total_counts_parquet_path = os.path.join(path, 'totalcounts.parquet')

        self.__identifier = identifier

    def read(self):
        if not os.path.exists(self.__data_parquet_path):
            raw_input_df = Application.get_or_create() \
                .read.csv(self.__path, sep='\n').withColumnRenamed('_c0', 'Input')

            split_df = raw_input_df \
                .select(split('Input', '\t').alias('SplitInput')) \
                .select(element_at('SplitInput', 1).alias('Tokens'),
                        slice('SplitInput', 2, size('SplitInput')).alias('Data')) \
                .select(split('Tokens', ' ').alias('Tokens'), 'Data') \

            final_df = split_df.select('Tokens', transform('Data', lambda d: split(d, ',')).alias('Data')) \
                .select('Tokens', transform('Data', lambda x: x[0]).alias('Year'),
                        transform('Data', lambda x: x[1]).cast(ArrayType(LongType())).alias('Occurrences')) \
                .select('Tokens', map_from_arrays('Year', 'Occurrences').alias('Data')) \
                .withColumn('Id', monotonically_increasing_id()) \
                .select(['Id', 'Tokens', 'Data'])
            final_df = final_df.withColumn('IdPart', ceil(final_df['Id'] / 1000000))
            final_df.show(20)

            final_df \
                .write \
                .option('parquet.bloom.filter.enable#Id', 'true') \
                .parquet(self.__data_parquet_path, partitionBy='IdPart')

        corpus_df = Application.get_or_create().read.parquet(self.__data_parquet_path).withColumn('Corpus', lit(self.__identifier))

        if not os.path.exists(self.__gin_parquet_path):
            token_df = corpus_df \
                .select(explode('Tokens').alias('Token'), 'Id') \
                .groupBy('Token') \
                .agg(collect_list('Id').alias('Ids')) \
                .repartitionByRange('Token')

            token_df.write.parquet(self.__gin_parquet_path)

        gin_df = Application.get_or_create().read.parquet(self.__gin_parquet_path)

        if not os.path.exists(self.__total_counts_parquet_path):
            raw_tc_df = Application.get_or_create().read.csv(os.path.join(self.__path, 'totalcounts'), sep='\t')
            raw_tc_df = raw_tc_df.withColumn('Entries', array(raw_tc_df.columns)).select(explode('Entries').alias('Entries'))
            raw_tc_df = raw_tc_df.withColumn('Splits', split('Entries', ',')).where(size('Splits') == 4)
            raw_tc_df = raw_tc_df.withColumn('Year', raw_tc_df['Splits'][0].cast(LongType())).withColumn('Frequency', raw_tc_df['Splits'][1].cast(LongType()))
            raw_tc_df = raw_tc_df.select(['Year', 'Frequency'])

            raw_tc_df.write.parquet(self.__total_counts_parquet_path)

        tc_df = Application.get_or_create().read.parquet(self.__total_counts_parquet_path)

        return corpus_df, gin_df, tc_df


