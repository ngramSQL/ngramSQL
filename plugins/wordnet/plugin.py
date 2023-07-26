import os
from typing import Dict, Any

import pandas as pd
import streamlit as st
import wn as wn
from pyspark.sql.functions import array_contains, explode, lit, collect_list
from pyspark.sql.types import StringType, StructType, StructField, ArrayType
from st_aggrid import GridOptionsBuilder, AgGrid

from core.application import Application
from core.plugin import BasePlugin


class WordnetPlugin(BasePlugin):
    __CONFIG_PREFIX = 'wordnet'

    __WORDNETS_CONFIG_PREFIX = 'wordnets'

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.__config = Application.get_or_create().get_specific_config(self.__CONFIG_PREFIX)
        print(self.__config)

        self.__wordnets = {}

        for wordnet_id in self.__config[self.__WORDNETS_CONFIG_PREFIX]:
            print(wordnet_id)

            wn_interface = WordnetInterface(self, wordnet_id)
            synsets, words, topics = wn_interface.read()
            self.__wordnets.update({wordnet_id: wn_interface})

            wordnet_id_sql = wordnet_id.replace(':', '_').replace('.', '_')

            Application.get_or_create().register_dataframe(synsets, self, wordnet_id_sql + '_synsets', ['wn_synset'])
            Application.get_or_create().register_dataframe(words, self, wordnet_id_sql + '_words', ['wn_word'])
            Application.get_or_create().register_dataframe(topics, self, wordnet_id_sql + '_topics', ['topic'])

        #Application.get_or_create().add_tab('Wordnet')

    def init_gui(self):
        pass


class WordnetInterface:

    def __init__(self, plugin, wordnet_id: str) -> None:
        self.__plugin = plugin
        self.__parquet_path = f'./parquets/wordnet/{wordnet_id}'

        self.__synsets = None
        self.__words = None
        self.__hyponym_graph = None

        if not os.path.exists(self.__parquet_path):
            wn.download(wordnet_id)
            self.__wordnet = wn.Wordnet(wordnet_id)

            wn.download('cili:1.0')

    def read(self):
        if not os.path.exists(self.__parquet_path):
            synsets = [(str(s.id),
                        str(s.pos),
                        str(s.ili),
                        str(s.definition()),
                        s.examples(),
                        [str(s.id) for s in s.senses()],
                        [str(w.id) for w in s.words()],
                        [str(s.id) for s in s.hypernyms()],
                        [str(s.id) for s in s.hyponyms()],
                        [str(s.id) for s in s.holonyms()],
                        [str(s.id) for s in s.meronyms()],
                        ) for s in self.__wordnet.synsets()]

            synset_df = Application.get_or_create().create_dataframe(
                synsets,
                StructType([
                    StructField('SynsetId', StringType()),
                    StructField('PartOfSpeech', StringType()),
                    StructField('ILI', StringType()),
                    StructField('Definition', StringType()),
                    StructField('Examples', ArrayType(StringType())),
                    StructField('Senses', ArrayType(StringType())),
                    StructField('WordIds', ArrayType(StringType())),
                    StructField('Hypernyms', ArrayType(StringType())),
                    StructField('Hyponyms', ArrayType(StringType())),
                    StructField('Holonyms', ArrayType(StringType())),
                    StructField('Meronyms', ArrayType(StringType()))
                ]), self.__plugin, 'synsets', None)

            synset_df.write.parquet(os.path.join(self.__parquet_path, 'synsets'))
            self.__synsets = synset_df

            words = [(str(w.id),
                      str(w.lemma()),
                      [str(f) for f in w.forms()],
                      [str(s.id) for s in w.synsets()],
                      [str(d.id) for d in w.derived_words()]) for w in self.__wordnet.words()]

            words_df = Application.get_or_create().create_dataframe(
                words,
                StructType([
                    StructField('WordId', StringType()),
                    StructField('Lemma', StringType()),
                    StructField('Forms', ArrayType(StringType())),
                    StructField('SynsetIds', ArrayType(StringType())),
                    StructField('DerivedWordIds', ArrayType(StringType()))
                ]), self.__plugin, 'words', None
            )

            words_df.write.parquet(os.path.join(self.__parquet_path, 'words'))
            self.__words = words_df

            self.__hyponym_graph = self.__synsets.withColumnRenamed('SynsetId', 'src') \
                .select('src', explode('Hyponyms').alias('dst'))

            topics_df = self.create_topic_df()
            topics_df.to_parquet(os.path.join(self.__parquet_path, 'topics'))


        self.__synsets = Application.get_or_create().read.parquet(os.path.join(self.__parquet_path, 'synsets'))
        self.__words = Application.get_or_create().read.parquet(os.path.join(self.__parquet_path, 'words'))
        self.__topics = Application.get_or_create().read.parquet(os.path.join(self.__parquet_path, 'topics'))

        return self.__synsets, self.__words, self.__topics

    def create_topic_df(self):
        transitive_closure = self.__hyponym_graph.toPandas()

        old_size = len(transitive_closure.index)

        while True:
            print(old_size)

            df_1 = transitive_closure.rename(columns={'dst': 'id'})
            df_2 = transitive_closure.rename(columns={'src': 'id'})

            new_tc = pd.concat([df_1.merge(df_2, on='id')[['src', 'dst']], transitive_closure]).drop_duplicates()

            new_size = len(new_tc.index)
            print(new_size)

            if new_size == old_size:
                break
            else:
                transitive_closure = new_tc

                old_size = new_size

        tc_list_df: pd.DataFrame = transitive_closure.groupby('src')['dst'].apply(list).reset_index().rename(columns={'src': 'SynsetId', 'dst': 'Hyponyms'})

        print(tc_list_df)

        synsets = self.__synsets.toPandas().drop('Hyponyms', axis=1)
        words = self.__words.toPandas()

        resolved_df = tc_list_df.merge(synsets, on='SynsetId')[['WordIds', 'Hyponyms']]
        resolved_df = resolved_df.explode('WordIds').rename(columns={'WordIds': 'WordId'})[['WordId', 'Hyponyms']]
        resolved_df = resolved_df.merge(words, on='WordId')
        resolved_df = resolved_df.explode('Hyponyms').rename(columns={'Hyponyms': 'SynsetId'})[['Lemma', 'SynsetId']].drop_duplicates()
        resolved_df = resolved_df.merge(synsets, on='SynsetId')[['Lemma', 'WordIds']]
        resolved_df = resolved_df.explode('WordIds').rename(columns={'Lemma': 'Topic', 'WordIds': 'WordId'})
        resolved_df = resolved_df.merge(words, on='WordId')[['Topic', 'Forms']].explode('Forms').rename(columns={'Forms': 'Form'}).drop_duplicates()
        resolved_df = resolved_df.groupby('Topic')['Form'].apply(list).reset_index().rename(columns={'Form': 'Members'})

        return resolved_df

