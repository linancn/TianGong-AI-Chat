import datetime
import json
import os
import time
from datetime import datetime

import streamlit as st

import ui_config
import utils
from utils import (
    StreamHandler,
    chat_history_chain,
    check_password,
    func_calling_chain,
    main_chain,
    seach_uploaded_docs,
    search_arxiv_docs,
    search_internet,
    search_pinecone,
    search_wiki,
    xata_chat_history,
)

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["XATA_API_KEY"] = st.secrets["xata_api_key"]
os.environ["XATA_DATABASE_URL"] = st.secrets["xata_db_url"]

ui = ui_config.create_ui_from_config()
st.set_page_config(page_title=ui.page_title, layout="wide", page_icon=ui.page_icon)


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
            search_knowledge_base = st.toggle(
                ui.search_knowledge_base_checkbox_label, value=False
            )
            search_online = st.toggle(ui.search_internet_checkbox_label, value=False)
            search_wikipedia = st.toggle(
                ui.search_wikipedia_checkbox_label, value=False
            )
            search_arxiv = st.toggle(ui.search_arxiv_checkbox_label, value=False)

            search_docs = st.toggle(ui.search_docs_checkbox_label, value=False)

            search_docs_option = None

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

            top_k_mappings = {}

            top_k_mappings["False_False_False_False_None"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 0,
            }
            top_k_mappings["False_False_False_False_Combined"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 16,
            }
            top_k_mappings["False_False_False_True_None"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 16,
                "search_docs_top_k": 0,
            }
            top_k_mappings["False_False_False_True_Combined"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 8,
                "search_docs_top_k": 8,
            }
            top_k_mappings["False_False_True_False_None"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 16,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 0,
            }
            top_k_mappings["False_False_True_False_Combined"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 8,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 8,
            }
            top_k_mappings["False_False_True_True_None"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 8,
                "search_arxiv_top_k": 8,
                "search_docs_top_k": 0,
            }
            top_k_mappings["False_False_True_True_Combined"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 5,
                "search_arxiv_top_k": 5,
                "search_docs_top_k": 6,
            }
            top_k_mappings["False_True_False_False_None"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 4,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 0,
            }
            top_k_mappings["False_True_False_False_Combined"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 2,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 8,
            }
            top_k_mappings["False_True_False_True_None"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 2,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 8,
                "search_docs_top_k": 0,
            }
            top_k_mappings["False_True_False_True_Combined"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 2,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 4,
                "search_docs_top_k": 4,
            }
            top_k_mappings["False_True_True_False_None"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 2,
                "search_wikipedia_top_k": 8,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 0,
            }
            top_k_mappings["False_True_True_False_Combined"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 2,
                "search_wikipedia_top_k": 4,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 4,
            }
            top_k_mappings["False_True_True_True_None"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 2,
                "search_wikipedia_top_k": 4,
                "search_arxiv_top_k": 4,
                "search_docs_top_k": 0,
            }
            top_k_mappings["False_True_True_True_Combined"] = {
                "search_knowledge_base_top_k": 0,
                "search_online_top_k": 1,
                "search_wikipedia_top_k": 4,
                "search_arxiv_top_k": 4,
                "search_docs_top_k": 4,
            }
            top_k_mappings["True_False_False_False_None"] = {
                "search_knowledge_base_top_k": 16,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 0,
            }
            top_k_mappings["True_False_False_False_Combined"] = {
                "search_knowledge_base_top_k": 8,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 8,
            }
            top_k_mappings["True_False_False_True_None"] = {
                "search_knowledge_base_top_k": 8,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 8,
                "search_docs_top_k": 0,
            }
            top_k_mappings["True_False_False_True_Combined"] = {
                "search_knowledge_base_top_k": 5,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 5,
                "search_docs_top_k": 6,
            }
            top_k_mappings["True_False_True_False_None"] = {
                "search_knowledge_base_top_k": 8,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 8,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 0,
            }
            top_k_mappings["True_False_True_False_Combined"] = {
                "search_knowledge_base_top_k": 5,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 5,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 6,
            }
            top_k_mappings["True_False_True_True_None"] = {
                "search_knowledge_base_top_k": 6,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 5,
                "search_arxiv_top_k": 5,
                "search_docs_top_k": 0,
            }
            top_k_mappings["True_False_True_True_Combined"] = {
                "search_knowledge_base_top_k": 4,
                "search_online_top_k": 0,
                "search_wikipedia_top_k": 4,
                "search_arxiv_top_k": 4,
                "search_docs_top_k": 4,
            }
            top_k_mappings["True_True_False_False_None"] = {
                "search_knowledge_base_top_k": 8,
                "search_online_top_k": 2,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 0,
            }
            top_k_mappings["True_True_False_False_Combined"] = {
                "search_knowledge_base_top_k": 8,
                "search_online_top_k": 1,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 4,
            }
            top_k_mappings["True_True_False_True_None"] = {
                "search_knowledge_base_top_k": 6,
                "search_online_top_k": 1,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 6,
                "search_docs_top_k": 0,
            }
            top_k_mappings["True_True_False_True_Combined"] = {
                "search_knowledge_base_top_k": 4,
                "search_online_top_k": 1,
                "search_wikipedia_top_k": 0,
                "search_arxiv_top_k": 4,
                "search_docs_top_k": 4,
            }
            top_k_mappings["True_True_True_False_None"] = {
                "search_knowledge_base_top_k": 6,
                "search_online_top_k": 1,
                "search_wikipedia_top_k": 6,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 0,
            }
            top_k_mappings["True_True_True_False_Combined"] = {
                "search_knowledge_base_top_k": 4,
                "search_online_top_k": 1,
                "search_wikipedia_top_k": 4,
                "search_arxiv_top_k": 0,
                "search_docs_top_k": 4,
            }
            top_k_mappings["True_True_True_True_None"] = {
                "search_knowledge_base_top_k": 4,
                "search_online_top_k": 1,
                "search_wikipedia_top_k": 4,
                "search_arxiv_top_k": 4,
                "search_docs_top_k": 0,
            }
            top_k_mappings["True_True_True_True_Combined"] = {
                "search_knowledge_base_top_k": 3,
                "search_online_top_k": 1,
                "search_wikipedia_top_k": 3,
                "search_arxiv_top_k": 3,
                "search_docs_top_k": 3,
            }

            current_top_k_mappings = f"{search_knowledge_base}_{search_online}_{search_wikipedia}_{search_arxiv}_{search_docs_option}"

            top_k_values = top_k_mappings.get(current_top_k_mappings)

            # override search_docs_top_k if search_docs_option is isolated
            if top_k_values is None:
                search_knowledge_base_top_k = 0
                search_online_top_k = 0
                search_wikipedia_top_k = 0
                search_arxiv_top_k = 0
                search_docs_top_k = 16
            else:
                search_knowledge_base_top_k = top_k_values.get(
                    "search_knowledge_base_top_k", 0
                )
                search_online_top_k = top_k_values.get("search_online_top_k", 0)
                search_wikipedia_top_k = top_k_values.get("search_wikipedia_top_k", 0)
                search_arxiv_top_k = top_k_values.get("search_arxiv_top_k", 0)
                search_docs_top_k = top_k_values.get("search_docs_top_k", 0)

        st.divider()

        col_newchat, col_delete = st.columns([1, 1])
        with col_newchat:
            new_chat = st.button(
                ui.sidebar_newchat_button_label, use_container_width=True
            )
        if new_chat:
            st.session_state.clear()
            st.rerun()

        with col_delete:
            delete_chat = st.button(
                ui.sidebar_delete_button_label, use_container_width=True
            )
        if delete_chat:
            utils.delete_chat_history(st.session_state["selected_chat_id"])
            st.session_state.clear()
            st.rerun()

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

    @utils.enable_chat_history
    def main():
        user_query = st.chat_input(placeholder=ui.chat_human_placeholder)

        if user_query:
            # chat_history.add_user_message(user_query)
            utils.display_msg(user_query, "user")
            st.session_state["xata_history"].add_user_message(user_query)

            chat_history_response = chat_history_chain()(
                {"input": st.session_state["xata_history"].messages[-6:]},
            )

            chat_history_recent = chat_history_response["text"]

            func_calling_response = func_calling_chain().run(chat_history_recent)

            query = func_calling_response.get("query")
            arxiv_query = func_calling_response.get("arxiv_query")

            try:
                created_at = json.loads(func_calling_response.get("created_at", None))
            except TypeError:
                created_at = None

            docs_response = []
            docs_response.extend(
                search_pinecone(query, created_at, top_k=search_knowledge_base_top_k)
            )
            docs_response.extend(search_internet(query, top_k=search_online_top_k))
            docs_response.extend(search_wiki(query, top_k=search_wikipedia_top_k))
            docs_response.extend(
                search_arxiv_docs(arxiv_query, top_k=search_arxiv_top_k)
            )
            docs_response.extend(seach_uploaded_docs(query, top_k=search_docs_top_k))

            input = f"""Provide a clear, well-organized, and critically analyzed respond to "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give detailed corresponding sources. Current conversation history:"{chat_history_recent}"""

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
                    st.rerun()

    if __name__ == "__main__":
        main()
