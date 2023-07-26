import math
import time
from collections import defaultdict

import pandas as pd
import streamlit as st
from pyspark.sql.functions import explode
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.express as px


class MainFrame:
    __instance = None

    def __init__(self, app):
        print('init mainframe')

        st.title('ngramSQL')

        self.__app = app

        self.__tab_names = self.__app.get_tabs()
        self.__tabs = st.tabs(self.__tab_names)
        self.__tab_dict = {}

        for i in range(len(self.__tab_names)):
            name = self.__tab_names[i]
            tab = self.__tabs[i]

            self.__tab_dict.update({name: tab})

        #####

        self.__explore_container = self.__tab_dict['Explore'].container()

        self.__corpus_dropdown = self.__explore_container.multiselect('Corpus Selection',
                                                                      self.__app.get_corpora(),
                                                                      max_selections=1)

        if self.__corpus_dropdown:
            self.__create_plot_expander()

            #self.__create_correlation_expander()

        #####

        self.__sql_container = self.__tab_dict['SQL'].container()

        self.__query_container = self.__sql_container.container()

        left_col, right_col = self.__query_container.columns([11, 1])
        self.__query_input = left_col.text_input('SQL Query')
        persist_result = False #right_col.button('Save Result')

        if self.__query_input:
            result = self.__app.execute_sql(self.__query_input)

            self.__sql_container.dataframe(result,
                                           use_container_width=True)

            if persist_result:
                result.write.parquet(f'./results/{time.time()}')

        #####

    def __create_plot_expander(self):
        self.__plot_expander = self.__explore_container.expander('Plot', expanded=False)

        self.__list_column, self.__plot_column = self.__plot_expander.columns([1, 3])

        num_results = 100
        data = self.__load_corpus_list(num_results)

        if data is not None:
            with self.__list_column:
                gb = GridOptionsBuilder.from_dataframe(data)
                gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
                gb.configure_selection('multiple')
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
                if len(data) == num_results:
                    st.warning(f'Only the first {num_results:,} results were loaded!', icon="⚠️")

                plot_relative = st.checkbox('plot relative frequencies')

                selected = self.__corpus_ngrams.selected_rows

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

    def __create_correlation_expander(self):
        self.__correlation_expander = self.__explore_container.expander('Topic Correlation', expanded=False)

        with self.__correlation_expander:
            topic_dfs = self.__app.get_dataframes_by_type('topic')

            wordnet_dropdown = st.multiselect('Topic Detector Selection', [x[0] for x in topic_dfs])

            if not wordnet_dropdown:
                return

            left, right = st.columns([1, 1])

            with left:
                first_query = st.text_input('First list of words')

            with right:
                second_query = st.text_input('Second list of words')

            st.dataframe(topic_dfs[0][1].limit(10), use_container_width=True)

    def __get_tc_dict(self, corpus_id):
        tc_df = self.__app.get_or_create().get_corpus(corpus_id)[2]

        to_return = defaultdict(int)
        for row in tc_df.collect():
            to_return.update({row['Year']: row['Frequency']})

        return to_return

    def __load_corpus_list(self, num_results):
        df_to_show = None

        if self.__corpus_dropdown:
            self.__corpus_ngram_search = self.__list_column.text_input('Token Search')

        for corpus_name in self.__corpus_dropdown:
            corpus, gin, tc = self.__app.get_or_create().get_corpus(corpus_name)

            relevant_ngrams = None
            if self.__corpus_ngram_search:
                relevant_ids = gin.where(gin['Token'].like(self.__corpus_ngram_search)).select(
                    explode('Ids').alias('Id'))
                relevant_id_list = [x['Id'] for x in relevant_ids.head(num_results)]

                # new_relevant_ngrams = relevant_ids.hint('broadcast').join(corpus, on=relevant_ids['Id'] == corpus['Id']).select(['Tokens', 'Data', 'Corpus'])
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

        # if df_to_show:
        #    df_to_show.cache()

        df_to_show.cache()
        # head() is MUCH faster than limit()!
        df_to_show_head = df_to_show.head(num_results)
        # df_to_show_head = df_to_show_head.persist()

        data = pd.DataFrame([r.asDict() for r in df_to_show_head])

        return data

    def get_tab(self, tab_name):
        return self.__tab_dict[tab_name]
