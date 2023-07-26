import math
from collections import defaultdict
from datetime import time
from typing import Dict, Any

import pandas as pd
import streamlit as st
from pyspark.sql.functions import explode
from st_aggrid import GridOptionsBuilder, AgGrid

import plotly.express as px

from core.application import Application
from core.plugin import BasePlugin


class DemoPlugin(BasePlugin):
    __TAB_NAME = 'Guided Tour'

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        if Application.get_or_create():
            Application.get_or_create().add_tab(self.__TAB_NAME)

    def init_gui(self):
        tab = Application.get_or_create().get_tab(self.__TAB_NAME)

        with tab:
            st.header('A Guided Tour of ngramSQL')

            st.markdown('''This demonstration of ngramSQL is meant to accompany our paper 
            _The ngramSQL Query System: Simplified Analysis of Large N-Gram Corpora_, submitted to ICADL 2023.
            ngramSQL offers the ability to plot frequency series of specific n-grams, similar to the _Google Books Ngram Viewer_.
            However, wildcards can be used more freely: The SQL wildcards **_** and **%** are supported, 
            matching one or several characters, respectively.
            The very flexible **\***-wildcard is not implemented in this first demo due to resource constraints.
            The corpus used for this demo is called *common_english* and contains the 100,000 most frequent 1-grams
            of the *Google Books Ngram Corpus*.
            ''')

            self.__plot_expander = st.expander(label='Plotting Frequencies', expanded=True)
            with self.__plot_expander:
                self.__corpus_dropdown = st.multiselect('Corpus Selection',
                                                        Application.get_or_create().get_corpora(),
                                                        default=list(Application.get_or_create().get_corpora())[0],
                                                        key='guided_tour_corpus_select')
                if self.__corpus_dropdown:
                    self.__fill_plot_expander()

            st.markdown('''The unique feature of ngramSQL however is its ability to handle arbitrary
            SQL queries against a relational representation of n-gram corpora.
            For this demo, the free formulation of queries is disabled, but will be made possible in locally hosted instances
            of ngramSQL.''')

            with st.expander(label='SQL Queries', expanded=True):
                queries = ['SHOW TABLES',
                           'SELECT * FROM common_english LIMIT 10',
                           'SELECT * FROM oewn_2021_synsets LIMIT 10',
                           "SELECT * FROM oewn_2021_topics WHERE Topic = 'philosophy'",
                           "SELECT Tokens, Data FROM (SELECT * FROM oewn_2021_topics WHERE Topic = 'philosophy') JOIN common_english WHERE array_contains(Members, Tokens[0]) ORDER BY Data[1965] DESC"]

                for query in queries:
                    query_col, result_col = st.columns([2, 5])

                    with query_col:
                        st.text_input(query, placeholder=query, label_visibility='collapsed', disabled=True)
                    with result_col:
                        result = Application.get_or_create().execute_sql(query)

                        st.dataframe(result, use_container_width=True)

    def __load_corpus_list(self, num_results):
        df_to_show = None

        if self.__corpus_dropdown:
            self.__corpus_ngram_search = self.__list_column.text_input('Token Search', value='%libr_r%')

        for corpus_name in self.__corpus_dropdown:
            corpus, gin, tc = Application.get_or_create().get_corpus(corpus_name)

            relevant_ngrams = None
            if self.__corpus_ngram_search:
                relevant_ids = gin.where(gin['Token'].like(self.__corpus_ngram_search)).select(
                    explode('Ids').alias('Id'))
                relevant_id_list = [x['Id'] for x in relevant_ids.head(num_results)]

                new_relevant_ngrams = corpus.where(corpus['Id'].isin(relevant_id_list))

                if not relevant_ngrams:
                    relevant_ngrams = new_relevant_ngrams
                else:
                    relevant_ngrams = relevant_ngrams.union(new_relevant_ngrams)
            else:
                relevant_ngrams = corpus

            if not df_to_show:
                df_to_show = relevant_ngrams
            else:
                df_to_show = df_to_show.union(relevant_ngrams)

        df_to_show.cache()
        # head() is MUCH faster than limit()!
        df_to_show_head = df_to_show.head(num_results)

        data = pd.DataFrame([r.asDict() for r in df_to_show_head])

        return data

    def __get_tc_dict(self, corpus_id):
        tc_df = Application.get_or_create().get_corpus(corpus_id)[2]

        to_return = defaultdict(int)
        for row in tc_df.collect():
            to_return.update({row['Year']: row['Frequency']})

        return to_return

    def __fill_plot_expander(self):
        self.__list_column, self.__plot_column = self.__plot_expander.columns([1, 3])

        num_results = 100
        data = self.__load_corpus_list(num_results)

        if data is not None:
            with self.__list_column:
                gb = GridOptionsBuilder.from_dataframe(data)
                gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
                gb.configure_selection('multiple')
                gb.configure_selection(pre_selected_rows=list(range(3)))
                gb.configure_column('Data', hide=True)

                custom_css = {
                    '.ag-theme-streamlit': {'--ag-background-color': '#dfdfdf',
                                            '--ag-odd-row-background-color': '#cecece',
                                            '--ag-foreground-color': '#000',
                                            '--ag-alpine-active-color': '#009682',
                                            '--ag-grid-size': '4px',
                                            '--ag-header-background-color': '#f8f9fb',
                                            '--ag-borders': 'solid 0.5px',
                                            '--ag-border-color': '#eaeaeb',
                                            '--ag-cell-horizontal-border': 'solid #eaeaeb',
                                            '--ag-header-foreground-color': '#7f838a',
                                            '--ag-font-family': '"Source Sans Pro"',
                                            '--ag-font-size': '9.5pt',
                                            '--ag-subheader-background-color': '#fff',
                                            '--ag-range-selection-border-color': '#009682',
                                            '--ag-subheader-toolbar-background-color': 'hsla(0,0%,100%,.5)',
                                            '--ag-selected-row-background-color': 'rgba(0,152,130,.1)',
                                            '--ag-row-hover-color': 'rgba(0,152,130,.1)',
                                            '--ag-column-hover-color': 'rgba(0,152,130,.1)',
                                            '--ag-chip-background-color': 'rgba(49,51,63,.07)',
                                            '--ag-input-disabled-background-color': 'hsla(240,2%,92%,.15)',
                                            '--ag-input-disabled-border-color': 'hsla(240,2%,92%,.3)',
                                            '--ag-disabled-foreground-color': 'rgba(49,51,63,.5)',
                                            '--ag-input-focus-border-color': 'rgba(0,152,130,.4)',
                                            '--ag-modal-overlay-background-color': 'hsla(0,0%,100%,.66)',
                                            '--ag-range-selection-background-color': 'rgba(0,152,130,.2)',
                                            '--ag-range-selection-background-color-2': 'rgba(0,152,130,.36)',
                                            '--ag-range-selection-background-color-3': 'rgba(0,152,130,.488)',
                                            '--ag-range-selection-background-color-4': 'rgba(0,152,130,.59)',
                                            '--ag-header-column-separator-color': 'hsla(240,2%,92%,.5)',
                                            '--ag-header-column-resize-handle-color': 'hsla(240,2%,92%,.5)',
                                            }
                }

                self.__corpus_ngrams = AgGrid(data, gb.build(),
                                              fit_columns_on_grid_load=True,
                                              theme='streamlit',
                                              custom_css=custom_css)
                if len(data) == 0:
                    st.warning(f'No results found for this input!', icon="⚠️")
                if len(data) == num_results:
                    st.warning(f'Only the first {num_results:,} results were loaded!', icon="⚠️")

                plot_relative = st.checkbox('plot relative frequencies')

                selected = self.__corpus_ngrams.selected_rows
                print(selected)

                min_year = math.inf
                max_year = -math.inf
                col_names = ['Year']
                to_plot = []

                for selected_row in selected:
                    tokens = selected_row['Tokens']
                    data = selected_row['Data']
                    corpus = selected_row['Corpus']

                    min_year = min(min_year, min([int(k) for k in data.keys()]))
                    max_year = max(max_year, max([int(k) for k in data.keys()]))

                    col_names.append(f'{str(tokens)}_{corpus}')

                    to_plot.append(data)

                if not (min_year == math.inf and max_year == -math.inf):
                    tc_dict = self.__get_tc_dict(self.__corpus_dropdown[0])

                    tuples = []

                    for year in range(int(min_year), (max_year + 1)):
                        new_tuple = [int(year), ]

                        for data_map in to_plot:

                            if str(year) in data_map:
                                if plot_relative:
                                    new_tuple.append(data_map[str(year)] / tc_dict[year])
                                else:
                                    new_tuple.append(data_map[str(year)])
                            else:
                                new_tuple.append(0)

                        tuples.append(tuple(new_tuple))

                    if tuples:
                        self.__plot = self.__plot_column.plotly_chart(
                            px.line(pd.DataFrame(tuples, columns=col_names), x='Year', y=col_names[1:]),
                            use_container_width=True)
