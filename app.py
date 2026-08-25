"""Cougar Career Agent — application entry point.

Owns page configuration and navigation. The three pages are:

* **Home** - what this is and whether the machine is configured
* **Learn the steps** - seven runnable steps that build up to the agent
* **Full demo** - the complete Job Hunter agent from the workshop script

Run it with::

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Cougar Career Agent",
    page_icon="🎓",
    layout="wide",
)

PAGES = [
    st.Page("pages/0_Home.py", title="Home", icon="🎓", default=True),
    st.Page("pages/1_Learn_the_Steps.py", title="Learn the steps", icon="📚"),
    st.Page("pages/2_Full_Demo.py", title="Full demo", icon="🎯"),
]

st.navigation(PAGES).run()
