import datetime
import json
import time
import uuid
from datetime import datetime

import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from streamlit.web.server.websocket_headers import _get_websocket_headers

import ui_config
import utils
import wix_oauth as wix_oauth
from sensitivity_checker import check_text_sensitivity
from top_k_mappings import top_k_mappings
from utils import (
    StreamHandler,
    chat_history_chain,
    check_password,
    delete_chat_history,
    fetch_chat_history,
    func_calling_chain,
    get_faiss_db,
    initialize_messages,
    main_chain,
    random_email,
    search_arxiv_docs,
    search_internet,
    search_pinecone,
    search_uploaded_docs,
    search_wiki,
    xata_chat_history,
)

ui = ui_config.create_ui_from_config()
st.set_page_config(page_title=ui.page_title, layout="wide", page_icon=ui.page_icon)

# CSS style injection
st.markdown(
    ui.page_markdown,
    unsafe_allow_html=True,
)

if "state" not in st.session_state:
    st.session_state["state"] = str(uuid.uuid4()).replace("-", "")
if "code_verifier" not in st.session_state:
    st.session_state["code_verifier"] = str(uuid.uuid4()).replace("-", "")

if "username" not in st.session_state or st.session_state["username"] is None:
    if st.secrets["wix_oauth"] and "logged_in" not in st.session_state:
        try:
            (
                auth,
                st.session_state["username"],
                st.session_state["subsription"],
            ) = wix_oauth.check_wix_oauth()
        except:
            pass
    elif st.secrets["anonymous_allowed"]:
        st.session_state["username"] = random_email()
        auth = True
    elif not st.secrets["anonymous_allowed"]:
        if ui.need_fixed_passwd is True:
            auth = check_password()
            if auth:
                st.session_state["username"] = random_email()
        elif ui.need_fixed_passwd is False:
            auth = False
            st.session_state["username"] = _get_websocket_headers().get(
                "Username", None
            )
            if st.session_state["username"] is not None:
                auth = True

try:
    if auth:
        st.session_state["logged_in"] = True
except:
    pass

if "logged_in" in st.session_state:
    try:
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

            if "subsription" in st.session_state:
                st.markdown(
                    ui.sidebar_welcome_text.format(
                        username=st.session_state["username"].split("@")[0],
                        subscription=st.session_state["subsription"],
                    )
                )

            with st.expander(ui.sidebar_expander_title, expanded=True):
                search_knowledge_base = st.toggle(
                    ui.search_knowledge_base_checkbox_label, value=False
                )
                search_online = st.toggle(
                    ui.search_internet_checkbox_label, value=False
                )
                search_wikipedia = st.toggle(
                    ui.search_wikipedia_checkbox_label, value=False
                )
                search_arxiv = st.toggle(ui.search_arxiv_checkbox_label, value=False)

                search_docs = st.toggle(
                    ui.search_docs_checkbox_label, value=False, disabled=True
                )

                # search_knowledge_base = True
                # search_online = st.toggle(ui.search_internet_checkbox_label, value=False)
                # search_wikipedia = False
                # search_arxiv = False
                # search_docs = False

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
                            st.session_state["faiss_db"] = get_faiss_db(uploaded_files)

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
                    search_wikipedia_top_k = top_k_values.get(
                        "search_wikipedia_top_k", 0
                    )
                    search_arxiv_top_k = top_k_values.get("search_arxiv_top_k", 0)
                    search_docs_top_k = top_k_values.get("search_docs_top_k", 0)

            st.markdown(body=ui.sidebar_instructions)

            st.divider()

            col_newchat, col_delete = st.columns([1, 1])
            with col_newchat:

                def init_new_chat():
                    keys_to_delete = [
                        "selected_chat_id",
                        "timestamp",
                        "first_run",
                        "messages",
                        "xata_history",
                        "uploaded_files",
                        "faiss_db",
                    ]
                    for key in keys_to_delete:
                        try:
                            del st.session_state[key]
                        except:
                            pass

                new_chat = st.button(
                    ui.sidebar_newchat_button_label,
                    use_container_width=True,
                    on_click=init_new_chat,
                )

            with col_delete:

                def delete_chat():
                    delete_chat_history(st.session_state["selected_chat_id"])
                    keys_to_delete = [
                        "selected_chat_id",
                        "timestamp",
                        "first_run",
                        "messages",
                        "xata_history",
                        "uploaded_files",
                        "faiss_db",
                    ]
                    for key in keys_to_delete:
                        try:
                            del st.session_state[key]
                        except:
                            pass

                delete_chat = st.button(
                    ui.sidebar_delete_button_label,
                    use_container_width=True,
                    on_click=delete_chat,
                )

            if "first_run" not in st.session_state:
                timestamp = time.time()
                st.session_state["timestamp"] = timestamp
            else:
                timestamp = st.session_state["timestamp"]

            try:  # fetch chat history from xata
                table_map = fetch_chat_history(st.session_state["username"])

                # add new chat to table_map
                table_map_new = {
                    str(timestamp): datetime.fromtimestamp(timestamp).strftime(
                        "%Y-%m-%d"
                    )
                    + " : "
                    + ui.sidebar_newchat_label
                }

                # Merge two dicts
                table_map = table_map_new | table_map
            except:  # if no chat history in xata
                table_map = {
                    str(timestamp): datetime.fromtimestamp(timestamp).strftime(
                        "%Y-%m-%d"
                    )
                    + " : "
                    + ui.sidebar_newchat_label
                }

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
                st.session_state["messages"] = initialize_messages(
                    st.session_state["xata_history"].messages
                )
    except:
        st.warning(ui.chat_error_message)

    @utils.enable_chat_history
    def main():
        if "xata_history_refresh" not in st.session_state:
            user_query = st.chat_input(placeholder=ui.chat_human_placeholder)
            if user_query:
                st.chat_message("user", avatar=ui.chat_user_avatar).markdown(user_query)
                st.session_state["messages"].append(
                    {"role": "user", "content": user_query}
                )
                human_message = HumanMessage(
                    content=user_query,
                    additional_kwargs={"id": st.session_state["username"]},
                )
                st.session_state["xata_history"].add_message(human_message)

                # check text sensitivity
                answer = check_text_sensitivity(user_query)["answer"]
                if answer is not None:
                    with st.chat_message("assistant", avatar=ui.chat_ai_avatar):
                        st.markdown(answer)
                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )
                        ai_message = AIMessage(
                            content=answer,
                            additional_kwargs={"id": st.session_state["username"]},
                        )
                        st.session_state["xata_history"].add_message(ai_message)
                else:
                    if len(st.session_state["messages"]) == 2:
                        chat_history_recent = ""
                        chat_history_recent_neat = ""
                    else:
                        current_message = st.session_state["messages"][-8:][1:][:-1]
                        for item in current_message:
                            item.pop("avatar", None)

                        chat_history_response = chat_history_chain()(
                            {"input": current_message},
                        )

                        chat_history_recent = chat_history_response["text"]
                        chat_history_recent_neat = chat_history_recent

                    if (
                        search_knowledge_base
                        or search_online
                        or search_wikipedia
                        or search_arxiv
                        or search_docs
                    ):
                        func_calling_response = func_calling_chain().run(
                            chat_history_recent
                            + " Latest message: "
                            + str(st.session_state["messages"][-1])
                        )

                        query = func_calling_response.get("query")
                        arxiv_query = func_calling_response.get("arxiv_query")

                        try:
                            created_at = json.loads(
                                func_calling_response.get("created_at", None)
                            )
                        except TypeError:
                            created_at = None

                        source = func_calling_response.get("source", None)

                        filters = {}
                        if created_at:
                            filters["created_at"] = created_at
                        if source:
                            filters["source"] = source

                        docs_response = []
                        docs_response.extend(
                            search_pinecone(
                                query=query,
                                filters=filters,
                                top_k=search_knowledge_base_top_k,
                            )
                        )
                        docs_response.extend(
                            search_internet(query, top_k=search_online_top_k)
                        )
                        docs_response.extend(
                            search_wiki(query, top_k=search_wikipedia_top_k)
                        )
                        docs_response.extend(
                            search_arxiv_docs(arxiv_query, top_k=search_arxiv_top_k)
                        )
                        docs_response.extend(
                            search_uploaded_docs(query, top_k=search_docs_top_k)
                        )

                        input = f"""You Must:
- Respond to "{user_query}" by using information from "{docs_response}" (if available) and your own knowledge to provide a logical, clear, and critically analyzed reply in the same language.
- Use the chat context from "{chat_history_recent}" (if available) to adjust the level of detail in your response.
- Employ bullet points selectively, where they add clarity or organization.
- Cite sources in main text using the Author-Date citation style where applicable.
- Provide a list of full references with hyperlinks, at the end for only the sources mentioned in the text.
- Use LaTeX quoted by '$' or '$$' within markdown to render mathematical formulas.

Avoid:
- Repeating information or including redundancies.
- Translating cited references into the query's language.
- Prefacing responses with any designation such as "AI:"."""
                        
                    else:
                        input = f"""Respond to "{user_query}". If "{chat_history_recent_neat}" is not empty, use it as chat context."""

                    with st.chat_message("assistant", avatar=ui.chat_ai_avatar):
                        st_cb = StreamHandler(st.empty())
                        response = main_chain()(
                            {"input": input},
                            callbacks=[st_cb],
                        )

                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": response["text"],
                            }
                        )
                        ai_message = AIMessage(
                            content=response["text"],
                            additional_kwargs={"id": st.session_state["username"]},
                        )
                        st.session_state["xata_history"].add_message(ai_message)

                if len(st.session_state["messages"]) == 3:
                    st.session_state["xata_history_refresh"] = True
                    st.rerun()
        else:
            user_query = st.chat_input(placeholder=ui.chat_human_placeholder)
            del st.session_state["xata_history_refresh"]

    if __name__ == "__main__":
        main()
