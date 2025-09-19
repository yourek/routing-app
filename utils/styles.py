import streamlit as st
import theme


def load_global_styles():
    with open("assets/global.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_theme():
    st.markdown(
        f"""
        <style>
            :root {{
                --primary-color: {theme.PRIMARY_COLOR};
                --secondary-color: {theme.SECONDARY_COLOR};
                --cancel-color: {theme.CANCEL_COLOR};
                --background-color: {theme.BACKGROUND_COLOR};
                --sidebar-bg-color: {theme.SIDEBAR_BG_COLOR};
                --text-color: {theme.TEXT_COLOR};
                --header-bg-color: {theme.HEADER_BG_COLOR};
                --header-text-color: {theme.HEADER_TEXT_COLOR};
                --border-color: {theme.BORDER_COLOR};
                --input-bg-color: {theme.INPUT_BG_COLOR};
                --navbar-height: {theme.NAVBAR_HEIGHT};
                --border-radius: {theme.BORDER_RADIUS};
                --padding: {theme.PADDING};
                --title-size: {theme.FONTS['title']};
                --body-size: {theme.FONTS['body']};
                --icon-size: {theme.FONTS['icon']};
            }}
        </style>
    """,
        unsafe_allow_html=True,
    )


def load_styles():
    load_theme()
    load_global_styles()
