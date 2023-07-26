import logging
import streamlit as st

from ngramsql.core.application import Application
from ngramsql.core.mainframe import MainFrame


def main() -> None:
    logging.basicConfig(level=logging.WARN)

    st.set_page_config(layout='wide')

    app = Application.get_or_create()
    main_frame = MainFrame(app)
    app.set_main_frame(main_frame)


if __name__ == '__main__':
    main()
