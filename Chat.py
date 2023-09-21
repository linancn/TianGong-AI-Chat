import datetime
import json
import os
import re
import time
from datetime import datetime

import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

import ui_config
import utils
from streaming import StreamHandler
from utils import search_pinecone, search_internet, seach_docs, func_calling_chain, chat_history_chain, main_chain, xata_chat_history

llm_model = st.secrets["llm_model"]
langchain_verbose = st.secrets["langchain_verbose"]

ui = ui_config.create_ui_from_config()

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["XATA_API_KEY"] = st.secrets["xata_api_key"]
os.environ["XATA_DATABASE_URL"] = st.secrets["xata_db_url"]

st.set_page_config(page_title=ui.page_title, layout="wide", page_icon=ui.page_icon)


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


if ui.need_passwd is False:
    auth = True
else:
    auth = check_password()


if auth:
    # 注入CSS style, 修改最上渐变条颜色
    st.markdown(
        ui.page_markdown,
        unsafe_allow_html=True,
    )

    # SIDEBAR
    with st.sidebar:
        st.markdown(
            ui.sidebar_markdown,
            unsafe_allow_html=True,
        )
        col_image, col_text = st.columns([1, 4])
        with col_image:
            st.image(ui.sidebar_image)
        with col_text:
            st.title(ui.sidebar_title)
        st.subheader(ui.sidebar_subheader)

        with st.expander(ui.sidebar_expander_title, expanded=False):
            search_online = st.toggle(ui.search_internet_checkbox_label, value=False)
            search_docs = st.toggle(ui.search_docs_checkbox_label, value=False)

            if search_docs:
                search_docs_option = st.radio(
                    label=ui.search_docs_options,
                    options=(
                        ui.search_docs_options_combined,
                        ui.search_docs_options_isolated,
                    ),
                    horizontal=True,
                )
                uploaded_files = st.file_uploader(
                    ui.sidebar_file_uploader_title,
                    accept_multiple_files=True,
                    type=None,
                )

                if uploaded_files != [] and uploaded_files != st.session_state.get(
                    "uploaded_files"
                ):
                    st.session_state["uploaded_files"] = uploaded_files
                    with st.spinner(ui.sidebar_file_uploader_spinner):
                        st.session_state["faiss_db"] = utils.get_faiss_db(
                            uploaded_files
                        )

        st.divider()

        col_newchat, col_delete = st.columns([1, 1])
        with col_newchat:
            new_chat = st.button(
                ui.sidebar_newchat_button_label, use_container_width=True
            )
        if new_chat:
            st.session_state.clear()
            st.experimental_rerun()

        with col_delete:
            delete_chat = st.button(
                ui.sidebar_delete_button_label, use_container_width=True
            )
        if delete_chat:
            utils.delete_chat_history(st.session_state["selected_chat_id"])
            st.session_state.clear()
            st.experimental_rerun()

        # fetch chat history from xata
        table_map = utils.fetch_chat_history()
        if "first_run" not in st.session_state:
            timestamp = time.time()
            st.session_state["timestamp"] = timestamp
        else:
            timestamp = st.session_state["timestamp"]

        # add new chat to table_map
        table_map_new = {
            str(timestamp): datetime.fromtimestamp(timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            + " - New Chat"
        }

        # Merge two dicts
        table_map = table_map_new | table_map

        # Get all keys from table_map into a list
        entries = list(table_map.keys())
        # Check if selected_chat_id exists in session_state, if not set default as the first entry
        if "selected_chat_id" not in st.session_state:
            st.session_state["selected_chat_id"] = entries[0]

        # Update the selectbox with the current selected_chat_id value
        current_chat_id = st.selectbox(
            label=ui.current_chat_title,
            label_visibility="collapsed",
            options=entries,
            format_func=lambda x: table_map[x],
            # index=entries.index(
            #     st.session_state["selected_chat_id"]
            # ),  # Use the saved value's index
        )

        # Save the selected value back to session state
        st.session_state["selected_chat_id"] = current_chat_id

        if "first_run" not in st.session_state:
            st.session_state["xata_history"] = xata_chat_history(
                _session_id=current_chat_id
            )
            st.session_state["first_run"] = True
        else:
            st.session_state["xata_history"] = xata_chat_history(
                _session_id=current_chat_id
            )
            st.session_state["messages"] = utils.initialize_messages(
                st.session_state["xata_history"].messages
            )

    class Basic:
        def __init__(self):
            self.openai_model = llm_model
            self.verbose = langchain_verbose

        @utils.enable_chat_history
        def main():
            # if "chat_history" not in st.session_state:
            #     st.session_state["chat_history"] = ChatMessageHistory()
            # else:
            #     chat_history = st.session_state["chat_history"]

            user_query = st.chat_input(placeholder=ui.chat_human_placeholder)

            if user_query:
                # chat_history.add_user_message(user_query)
                utils.display_msg(user_query, "user")
                st.session_state["xata_history"].add_user_message(user_query)

                if len(st.session_state["xata_history"].messages) <= 1:
                    func_calling_response = func_calling_chain().run(user_query)

                    query = func_calling_response.get("query")
                    try:
                        created_at = json.loads(
                            func_calling_response.get("created_at", None)
                        )
                    except TypeError:
                        created_at = None

                    if search_docs:
                        if search_docs_option == ui.search_docs_options_isolated:
                            docs_response = seach_docs(query, top_k=16)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources."""
                        elif search_docs_option == ui.search_docs_options_combinedd:
                            if search_online:
                                embedding_results = search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                internet_results = search_internet(query)
                                docs_response.extend(internet_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                            elif not search_online:
                                embedding_results = search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                    elif not search_docs:
                        if search_online:
                            embedding_results = search_pinecone(
                                query, created_at, top_k=16
                            )
                            internet_results = search_internet(query)
                            embedding_results.extend(internet_results)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                        elif not search_online:
                            embedding_results = search_pinecone(
                                query, created_at, top_k=16
                            )
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                else:
                    chat_history_response = chat_history_chain()(
                        {"input": st.session_state["xata_history"].messages[-7:]},
                    )
                    chat_history_summary = chat_history_response["text"]

                    func_calling_response = func_calling_chain().run(
                        chat_history_summary
                    )

                    query = func_calling_response.get("query")

                    try:
                        created_at = json.loads(
                            func_calling_response.get("created_at", None)
                        )
                    except TypeError:
                        created_at = None

                    if search_docs:
                        if search_docs_option == ui.search_docs_options_isolated:
                            docs_response = seach_docs(query, top_k=16)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources. Current conversation:"{chat_history_summary}"""
                        elif search_docs_option == ui.search_docs_options_combined:
                            if search_online:
                                embedding_results = search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                internet_results = search_internet(query)
                                docs_response.extend(internet_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""
                            elif not search_online:
                                embedding_results = search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""
                    elif not search_docs:
                        if search_online:
                            embedding_results = search_pinecone(
                                query, created_at, top_k=16
                            )
                            internet_results = search_internet(query)
                            embedding_results.extend(internet_results)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""
                        elif not search_online:
                            embedding_results = search_pinecone(
                                query, created_at, top_k=16
                            )
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""

                with st.chat_message("assistant", avatar=ui.chat_ai_avatar):
                    st_cb = StreamHandler(st.empty())
                    response = main_chain()(
                        {"input": input},
                        callbacks=[st_cb],
                    )

                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "avatar": ui.chat_ai_avatar,
                            "content": response["text"],
                        }
                    )
                    st.session_state["xata_history"].add_ai_message(response["text"])

                    if len(st.session_state["messages"]) == 3:
                        st.experimental_rerun()

    if __name__ == "__main__":
        obj = Basic()
        obj.main()
