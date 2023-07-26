import math
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import pandas
import pandas as pd
import plotly.express as px
from pyspark.sql.functions import lit, array, split, array_contains, collect_set, udf, explode, flatten, broadcast, \
    collect_list
from pyspark.sql.types import StringType, ArrayType, IntegerType

from core.application import Application
from core.plugin import BasePlugin


class NgramListsPlugin(BasePlugin):
    __CONFIG_PREFIX = 'ngramlists'
    __CONFIG_PREFIX_LISTS = 'lists'

    __TAB_NAME = 'N-Gram Lists'

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.__config = Application.get_or_create().get_specific_config(self.__CONFIG_PREFIX)

        self.__dist_measures = {'Euclidean Distance': (self.__euclidean, 1),
                                'Pearson Correlation': (self.__pearson, -1)}

        self.__lists = {}
        for path in self.__config[self.__CONFIG_PREFIX_LISTS]:
            with open(path, 'r') as file:
                word_list = []

                for line in file:
                    word_list.append(line.strip('\n'))

            self.__lists.update({path: word_list})

        if Application.get_or_create():
            Application.get_or_create().add_tab(self.__TAB_NAME)

    def __get_tab(self):
        return Application.get_or_create().get_tab(self.__TAB_NAME)

    def init_gui(self):
        tab = self.__get_tab()

        outer_container = tab.container()
        self.__agg_expander(outer_container)
        self.__knn_expander(outer_container)
        self.__change_expander(outer_container)
        self.__join_expander(outer_container)

    def __plot(self, container, time_series_dict):
        pass

    def __time_series_dict_to_pandas(self, start_year, end_year, time_series_dict, tc_dict=None):
        df_list = []
        for year in range(int(start_year), int(end_year) + 1):
            values = []

            for ngram in time_series_dict.keys():
                if tc_dict:
                    if tc_dict[year] != 0:
                        values.append(time_series_dict[ngram][year] / tc_dict[year])
                else:
                    values.append(time_series_dict[ngram][year])

            df_list.append(values)

        to_plot_df = pandas.DataFrame(df_list, columns=list(time_series_dict.keys()),
                                      index=range(int(start_year), int(end_year) + 1))

        return to_plot_df

    @staticmethod
    def __intersection_udf(lists):
        if len(lists) == 1:
            return lists[0]

        intersection_list = [value for value in lists[0] if value in NgramListsPlugin.__intersection_udf(lists[1:])]
        return intersection_list

    def __get_time_series_dict(self, corpus, ngram_list, tc_dict=None):
        corpus_df, corpus_gin_df, tc_df = Application.get_or_create().get_corpus(corpus)

        intersection_udf = udf(self.__intersection_udf, ArrayType(IntegerType()))

        if isinstance(ngram_list, str):
            actual_ngram_list = self.__lists[ngram_list]
        else:
            actual_ngram_list = ngram_list

        ngram_df = Application.get_or_create().create_dataframe(actual_ngram_list, StringType(), self, 'ngram', 'tmp')
        ngram_df.show()
        ngram_df = ngram_df.withColumnRenamed('value', 'NGram')
        ngram_df = ngram_df.select(['NGram', split(ngram_df['NGram'], ' ').alias('Tokens')])
        ngram_df = ngram_df.select(['NGram', explode('Tokens').alias('Token')])
        ngram_df.show()
        ngram_df = corpus_gin_df.join(broadcast(ngram_df), on=ngram_df['Token'] == corpus_gin_df['Token'])
        ngram_df.cache()
        ngram_df = ngram_df.groupBy('NGram').agg(flatten(collect_list('Ids'))).withColumnRenamed('flatten(collect_list(Ids))', 'Ids')
        ngram_df = ngram_df.select(['NGram', split(ngram_df['NGram'], ' ').alias('Tokens'), 'Ids'])
        ngram_df.show()
        ngram_df = ngram_df.groupBy('Tokens').agg(collect_set('Ids')).withColumnRenamed('collect_set(Ids)', 'IdSets')
        ngram_df = ngram_df.select('Tokens', flatten(ngram_df['IdSets']).alias('IdIntersection'))
        ngram_df = ngram_df.select('Tokens', explode('IdIntersection').alias('IdIntersection'))
        ngram_df = ngram_df.withColumnRenamed('Tokens', 'Tokens2')
        ngram_df.explain()
        ngram_df.show()
        ngram_df = corpus_df.join(broadcast(ngram_df), on=ngram_df['IdIntersection'] == corpus_df['Id'])
        ngram_df.cache()
        ngram_df = ngram_df.where(ngram_df['Tokens'] == ngram_df['Tokens2']).select(['Tokens', 'Data'])
        ngram_df.explain()
        ngram_df.cache()
        ngram_rows = ngram_df.collect()

        to_return = {}
        for ngram_row in ngram_rows:
            data = defaultdict(int)
            for k, v in ngram_row['Data'].items():
                if tc_dict:
                    if tc_dict[int(k)] != 0:
                        data.update({int(k): int(v) / tc_dict[int(k)]})
                else:
                    data.update({int(k): int(v)})

            to_return.update({' '.join(ngram_row['Tokens']): data})

        return to_return

    def __result_container(self, parent, min_year, max_year, time_series_dict, tc_dict=None, x_label='Year', y_label='Frequency'):
        outer = parent.container()

        plot_relative = tc_dict is not None

        if plot_relative:
            to_plot = self.__time_series_dict_to_pandas(min_year, max_year, time_series_dict, tc_dict)
        else:
            to_plot = self.__time_series_dict_to_pandas(min_year, max_year, time_series_dict)

        outer.plotly_chart(
            px.line(pd.DataFrame(to_plot)),
            use_container_width=True)

    @staticmethod
    def __corpus_selector(parent, key, max_selections=None):
        return parent.multiselect('Corpus Selection',
                                  Application.get_or_create().get_corpora(),
                                  key='ngram_lists.corpus_sel' + key,
                                  max_selections=max_selections)

    def __word_list_selector(self, parent, key, max_selections=None):
        word_lists = parent.multiselect('N-Gram List Selection',
                                        self.__lists,
                                        key='ngram_lists.lists_sel' + key,
                                        max_selections=5)

        if word_lists:
            word_lists_tuples = []
            word_lists_dict = {}

            for word_list in word_lists:
                word_lists_tuples.append((word_list, self.__lists[word_list]))
                word_lists_dict.update({word_list: self.__lists[word_list]})

            print(word_lists_tuples)
            parent.dataframe(pd.DataFrame(word_lists_tuples, columns=['List', 'N-Gram']), use_container_width=True)

            return word_lists_dict

    def __get_tc_dict(self, corpus_id):
        tc_df = Application.get_or_create().get_corpus(corpus_id)[2]

        to_return = defaultdict(int)
        for row in tc_df.collect():
            to_return.update({row['Year']: row['Frequency']})

        return to_return

    def __dist_measure(self, container):
        dist_measure = container.selectbox('Distance Measure', list(self.__dist_measures.keys()), label_visibility='collapsed')

        return self.__dist_measures[dist_measure]

    def __agg_expander(self, outer_container):
        agg_expander = outer_container.expander('Aggregation', False)
        left_col, right_col = agg_expander.columns([1, 2])

        corpora = self.__corpus_selector(left_col, 'agg', max_selections=1)
        word_lists = self.__word_list_selector(left_col, 'agg')

        disable_evaluation = (corpora is not None and word_lists is not None) and not (len(corpora) > 0 and len(word_lists) > 0)
        evaluate = left_col.button('Evaluate', disabled=disable_evaluation)

        if evaluate:
            time_series_dicts = [(word_list, self.__get_time_series_dict(corpora[0], word_list, self.__get_tc_dict(corpora[0]))) for word_list in word_lists]
            print(time_series_dicts)

            min_year = np.inf
            max_year = -np.inf

            aggregates = {}
            for list_name, time_series_dict in time_series_dicts:
                aggregate = defaultdict(int)

                for _, data in time_series_dict.items():
                    for year, entry in data.items():
                        aggregate[int(year)] += entry

                        if int(year) < min_year:
                            min_year = int(year)
                        if int(year) > max_year:
                            max_year = int(year)

                aggregates.update({list_name: aggregate})

            self.__result_container(right_col, min_year, max_year, aggregates)

    def __unpack_time_series_dict(self, ngram_dicts, start_year, end_year):
        time_series = {}

        for ngram, data_dict in ngram_dicts.items():
            data_list = []

            for i in range(int(start_year), int(end_year) + 1):
                if i in data_dict.keys():
                    data_list.append(data_dict[i])
                else:
                    data_list.append(0)

            time_series.update({ngram: data_list})

        return time_series

    def __euclidean(self, a, b):
        dist = 0

        for i in range(len(a)):
            dist += (a[i] - b[i])**2

        return math.sqrt(dist)

    def __pearson(self, a, b):
        return np.corrcoef(a, b)[0][1]

    def __compute_knn(self, k, target, candidates, distance_measure):
        print(candidates)

        dist_tuples = []

        for candidate in candidates:
            dist = distance_measure[0](target[1], candidate[1])

            dist_tuples.append((candidate[0], dist))

        dist_tuples = sorted(dist_tuples, key=lambda x: x[1] * distance_measure[1])

        return dist_tuples[:k]

    def __knn_expander(self, outer_container):
        knn_expander = outer_container.expander('Nearest Neighbors', False)
        left_col, right_col = knn_expander.columns([1, 2])

        corpora = self.__corpus_selector(left_col, 'knn', max_selections=1)
        lists = self.__word_list_selector(left_col, 'knn', max_selections=1)

        input_form = left_col.form('input_form_knn')
        target_ngram = input_form.text_input('Target N-Gram', label_visibility='collapsed', placeholder='Target N-Gram')

        inner_col_l, inner_col_r = input_form.columns([1, 1])
        start_year = inner_col_l.text_input('Start Year', label_visibility='collapsed', placeholder='Start Year')
        end_year = inner_col_r.text_input('End Year', label_visibility='collapsed', placeholder='End Year')

        number_nn = inner_col_l.text_input('Number of Neighbors', label_visibility='collapsed',
                                           placeholder='Number of Neighbors')
        dist_measure = self.__dist_measure(inner_col_r)

        evaluate = input_form.form_submit_button('Evaluate')

        if evaluate and not (target_ngram and start_year and end_year and number_nn and dist_measure):
            input_form.warning('Please fill in all the fields above.')
        elif evaluate and (target_ngram and start_year and end_year and number_nn and dist_measure) and len(corpora) > 0 and len(lists) > 0:
            ngram_dicts = self.__get_time_series_dict(corpora[0], list(lists.values())[0] + [target_ngram], tc_dict=self.__get_tc_dict(corpora[0]))

            candidates = self.__unpack_time_series_dict(ngram_dicts, start_year, end_year)
            target = (target_ngram, candidates[target_ngram])

            nbrs = self.__compute_knn(int(number_nn), target, list(candidates.items()), dist_measure)

            ngram_dicts_to_plot = {target_ngram: ngram_dicts[target[0]]}
            for nbr in nbrs:
                ngram_dicts_to_plot.update({str(nbr): ngram_dicts[nbr[0]]})

            to_plot_df = self.__time_series_dict_to_pandas(start_year, end_year, ngram_dicts_to_plot)

            #right_col.plotly_chart(px.line(to_plot_df), use_container_width=True)

            self.__result_container(right_col, start_year, end_year, ngram_dicts_to_plot)

    def __change_expander(self, outer_container):
        change_expander = outer_container.expander('Change', False)
        left_col, right_col = change_expander.columns([1, 2])

        corpora = self.__corpus_selector(left_col, 'change', max_selections=1)
        lists = self.__word_list_selector(left_col, 'change', max_selections=1)

        input_form = left_col.form('input_form_change')

        inner_col_l, inner_col_r = input_form.columns([1, 1])
        start_year = inner_col_l.text_input('Start Year', label_visibility='collapsed', placeholder='Start Year')
        end_year = inner_col_r.text_input('End Year', label_visibility='collapsed', placeholder='End Year')

        threshold = inner_col_l.text_input('Absolute Threshold', label_visibility='collapsed',
                                           placeholder='Absolute Threshold')
        num_results = inner_col_r.text_input('Number of Results', label_visibility='collapsed',
                                             placeholder='Number of Results')

        abs_rel = inner_col_r.radio('Absolute or Relative Change', ['absolute', 'relative'], label_visibility='collapsed')
        use_absolute_change = (abs_rel == 'absolute')

        evaluate = input_form.form_submit_button('Evaluate')

        if evaluate and (not (start_year and end_year and threshold and num_results) or len(corpora) <= 0 or len(lists) <= 0):
            input_form.warning('Please fill in all the fields above.')
        elif evaluate and (start_year and end_year and threshold and num_results) and len(corpora) > 0 and len(lists) > 0:
            time_series_dict = self.__get_time_series_dict(corpora[0], list(lists.values())[0])

            results = []
            for ngram, data in time_series_dict.items():
                val_start = data[int(start_year)]
                val_end = data[int(end_year)]

                if not (val_start >= int(threshold) and val_end >= int(threshold)):
                    continue

                if use_absolute_change:
                    change = abs(val_end - val_start)
                else:
                    change = val_end / val_start
                    if change < 1:
                        change = 1/change

                results.append((ngram, change))
            print(results)
            results = sorted(results, key=lambda x: -x[1])[:int(num_results)]
            print(results)

            result_dict = {}
            for ngram, change in results:
                result_dict.update({str(ngram) + ': ' + str(change): time_series_dict[ngram]})

            self.__result_container(right_col, start_year, end_year, result_dict)

    def __join_expander(self, outer_container):
        join_expander = outer_container.expander('Similarity Join', False)
        left_col, right_col = join_expander.columns([1, 2])

        corpora = self.__corpus_selector(left_col, 'sim', max_selections=1)
        lists = self.__word_list_selector(left_col, 'sim', max_selections=1)

        input_form = left_col.form('input_form_join')

        inner_col_l, inner_col_r = input_form.columns([1, 1])
        start_year = inner_col_l.text_input('Start Year', label_visibility='collapsed', placeholder='Start Year')
        end_year = inner_col_r.text_input('End Year', label_visibility='collapsed', placeholder='End Year')

        dist_measure = self.__dist_measure(inner_col_l)
        num_results = inner_col_r.text_input('Number of Results', label_visibility='collapsed',
                                             placeholder='Number of Results')

        evaluate = input_form.form_submit_button('Evaluate')

        if evaluate and (not (start_year and end_year and dist_measure and num_results) or len(corpora) <= 0 or len(lists) <= 0):
            input_form.warning('Please fill in all the fields above.')
        elif evaluate and (start_year and end_year and dist_measure and num_results) and len(corpora) > 0 and len(lists) > 0:
            time_series_dict = self.__get_time_series_dict(corpora[0], list(lists.values())[0], self.__get_tc_dict(corpora[0]))

            results = []
            unpacked_data = self.__unpack_time_series_dict(time_series_dict, start_year, end_year).items()

            for ngram_l, data_l in unpacked_data:
                for ngram_r, data_r in unpacked_data:
                    if ngram_l >= ngram_r:
                        continue

                    dist = dist_measure[0](data_l, data_r)

                    results.append((ngram_l, ngram_r, dist))

            results = sorted(results, key=lambda x: x[2] * dist_measure[1])[:int(num_results)]

            right_col.dataframe(pd.DataFrame(results, columns=['Left', 'Right', 'Similarity']), use_container_width=True)
