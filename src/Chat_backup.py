import datetime
import json
import time
from datetime import datetime
import requests

import streamlit as st
import streamlit.components.v1 as components
from langchain.schema import AIMessage, HumanMessage
from streamlit.web.server.websocket_headers import _get_websocket_headers

import ui_config
import utils
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

if "username" not in st.session_state:
    if st.secrets["anonymous_allowed"]:
        st.session_state["username"] = random_email()
    else:
        st.session_state["username"] = _get_websocket_headers().get(
            "Username", "unknown@unknown.com"
        )


# st.write(st.session_state["username"])


CLIENT_ID = st.secrets["wix_client_id"]
# REDIRECT_URI = st.secrets["wix_redirect_uri"]


def wix_get_access_token():
    request = requests.post(
        "https://www.wixapis.com/oauth2/token",
        headers={
            "content-type": "application/json",
        },
        json={"clientId": CLIENT_ID, "grantType": "anonymous"},
    )
    return request.json()["access_token"]


def wix_login(access_token: str, username: str, password: str):
    url = "https://www.wixapis.com/_api/iam/authentication/v2/login"

    headers = {
        "authorization": access_token,
        "content-type": "application/json",
    }

    data = {"loginId": {"email": username}, "password": password}

    response = requests.post(url, headers=headers, json=data)
    response_text = json.loads(response.text)

    if response_text["state"] == "SUCCESS":
        session_token = response_text["sessionToken"]
        respond = requests.post(
            "https://www.wixapis.com/_api/redirects-api/v1/redirect-session",
            headers={
                "authorization": access_token,
                "content-type": "application/json",
            },
            json={
                "auth": {
                    "authRequest": {
                        "clientId": CLIENT_ID,
                        "codeChallenge": "JNU5gZmEjgVL2eXfgSmUW3S2E202k2rkq4u3M_drdCY",
                        "codeChallengeMethod": "S256",
                        "responseMode": "web_message",
                        "responseType": "code",
                        "scope": "offline_access",
                        "state": "Z4dy7JM2S7n35VnBhdMeOQyXQW7UkE2Q1afdPLL419o",
                        "sessionToken": session_token,
                    }
                }
            },
        )

        url_response = respond.json()["redirectSession"]["fullUrl"]

        my_component = components.declare_component("my_component", url=url_response)

        # Send data to the frontend using named arguments.
        return_value = my_component(url=url_response)

        # `my_component`'s return value is the data returned from the frontend.
        st.write("Value = ", return_value)

        orders_response = requests.get(
            "https://www.wixapis.com/pricing-plans/v2/member/orders",
            headers={"authorization": session_token},
        )

    return response.json()


col_left, col_center, col_right = st.columns(3)

with col_center:
    with st.form(key="login_form"):
        username = st.text_input("username")
        password = st.text_input("password", type="password")
        submit = st.form_submit_button("Login")

if submit:
    wix_access_token = wix_get_access_token()
    wix_login_response = wix_login(
        access_token=wix_access_token, username=username, password=password
    )


st.stop()

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

        with st.expander(ui.sidebar_expander_title, expanded=True):
            search_knowledge_base = st.toggle(
                ui.search_knowledge_base_checkbox_label, value=False
            )
            search_online = st.toggle(ui.search_internet_checkbox_label, value=True)
            search_wikipedia = st.toggle(
                ui.search_wikipedia_checkbox_label, value=False
            )
            search_arxiv = st.toggle(ui.search_arxiv_checkbox_label, value=False)

            search_docs = st.toggle(ui.search_docs_checkbox_label, value=False)

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
                search_wikipedia_top_k = top_k_values.get("search_wikipedia_top_k", 0)
                search_arxiv_top_k = top_k_values.get("search_arxiv_top_k", 0)
                search_docs_top_k = top_k_values.get("search_docs_top_k", 0)

        st.markdown(body=ui.sidebar_instructions)

        st.divider()

        col_newchat, col_delete = st.columns([1, 1])
        with col_newchat:
            new_chat = st.button(
                ui.sidebar_newchat_button_label, use_container_width=True
            )
        if new_chat:
            # avoid rerun for new random email,no use clear()
            del st.session_state["selected_chat_id"]
            del st.session_state["timestamp"]
            del st.session_state["first_run"]
            del st.session_state["messages"]
            del st.session_state["xata_history"]
            try:
                del st.session_state["uploaded_files"]
            except:
                pass
            try:
                del st.session_state["faiss_db"]
            except:
                pass
            st.rerun()

        with col_delete:
            delete_chat = st.button(
                ui.sidebar_delete_button_label, use_container_width=True
            )
        if delete_chat:
            delete_chat_history(st.session_state["selected_chat_id"])
            # avoid rerun for new random email, no use clear()
            del st.session_state["selected_chat_id"]
            del st.session_state["timestamp"]
            del st.session_state["first_run"]
            del st.session_state["messages"]
            del st.session_state["xata_history"]
            try:
                del st.session_state["uploaded_files"]
            except:
                pass
            try:
                del st.session_state["faiss_db"]
            except:
                pass
            st.rerun()

        if "first_run" not in st.session_state:
            timestamp = time.time()
            st.session_state["timestamp"] = timestamp
        else:
            timestamp = st.session_state["timestamp"]

        try:  # fetch chat history from xata
            table_map = fetch_chat_history(st.session_state["username"])

            # add new chat to table_map
            table_map_new = {
                str(timestamp): datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                + " : New Chat"
            }

            # Merge two dicts
            table_map = table_map_new | table_map
        except:  # if no chat history in xata
            table_map = {
                str(timestamp): datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                + " : New Chat"
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

    @utils.enable_chat_history
    def main():
        user_query = st.chat_input(placeholder=ui.chat_human_placeholder)

        if user_query:
            st.chat_message("user", avatar=ui.chat_user_avatar).markdown(user_query)
            st.session_state["messages"].append({"role": "user", "content": user_query})
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
                chat_history_response = chat_history_chain()(
                    {"input": st.session_state["xata_history"].messages[-6:]},
                )

                chat_history_recent = chat_history_response["text"]

                func_calling_response = func_calling_chain().run(chat_history_recent)

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
                        query=query, filters=filters, top_k=search_knowledge_base_top_k
                    )
                )
                docs_response.extend(search_internet(query, top_k=search_online_top_k))
                docs_response.extend(search_wiki(query, top_k=search_wikipedia_top_k))
                docs_response.extend(
                    search_arxiv_docs(arxiv_query, top_k=search_arxiv_top_k)
                )
                docs_response.extend(
                    search_uploaded_docs(query, top_k=search_docs_top_k)
                )

                input = f""" You must:
use "{chat_history_recent}" to decide the response more concise or more detailed;
based on the "{docs_response}" and your own knowledge, provide a logical, clear, well-organized, and critically analyzed respond in the language of "{user_query}";
use bullet points only when necessary;
give in-text citations where relevant in Author-Date mode, NOT in Numeric mode;
list full reference information with hyperlinks at the end, for only those cited in the text.

You must not:
include any duplicate or redundant information;
translate reference to query's language;
return any prefix like "AI:"."""

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
                st.rerun()

    if __name__ == "__main__":
        main()
