import asyncio
import datetime
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
from utils import (
    StreamHandler,
    check_password,
    concurrent_search_service,
    count_chat_history,
    delete_chat_history,
    fetch_chat_history,
    func_calling_chain,
    get_begin_datetime,
    initialize_messages,
    main_chain,
    random_email,
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
            st.subheader("环境生态领域智能助手",help="Environment and Ecology Intelligent Assistant")

            if "subsription" in st.session_state:
                st.markdown(
                    ui.sidebar_welcome_text.format(
                        username=st.session_state["username"].split("@")[0],
                        subscription=st.session_state["subsription"],
                    )
                )

            base_model = st.radio(
                label="模型选择 / Model Selection",
                # label_visibility="collapsed",
                options=["ZHIPU 智谱", "BAIDU 百度"],
                horizontal=True,
                index=0,
                help="glm-4-plus / ernie-3.5-128k",
            )
            if base_model == "ZHIPU 智谱":
                api_key = st.secrets["openai_api_key_zhipu"]
                llm_model = st.secrets["llm_model_zhipu"]
                openai_api_base = st.secrets["openai_api_base_zhipu"]
            elif base_model == "BAIDU 百度":
                api_key = st.secrets["openai_api_key_baidu"]
                llm_model = st.secrets["llm_model_baidu"]
                openai_api_base = st.secrets["openai_api_base_baidu"]

            with st.expander(ui.sidebar_expander_title, expanded=True):
                if "search_option_disabled" not in st.session_state:
                    st.session_state["search_option_disabled"] = False

                search_sci = st.toggle(
                    ui.search_journal_paper_checkbox_label,
                    value=False,
                    disabled=st.session_state["search_option_disabled"],
                )
                search_report = st.toggle(
                    ui.search_report_checkbox_label,
                    value=False,
                    disabled=st.session_state["search_option_disabled"],
                )
                search_standard = st.toggle(
                    ui.search_standard_checkbox_label,
                    value=False,
                    disabled=st.session_state["search_option_disabled"],
                )

                search_patent = st.toggle(
                    ui.search_patent_checkbox_label,
                    value=False,
                    disabled=st.session_state["search_option_disabled"],
                )

                search_online = st.toggle(
                    ui.search_internet_checkbox_label,
                    value=False,
                    disabled=st.session_state["search_option_disabled"],
                )

                search_list = []
                if search_sci:
                    search_list.append("sci_search")
                if search_report:
                    search_list.append("report_search")
                if search_standard:
                    search_list.append("standard_search")
                if search_patent:
                    search_list.append("patent_search")
                if search_online:
                    search_list.append("internet_search")

                # if (
                #     "subsription" in st.session_state
                #     and st.session_state["subsription"] == "Elite"
                # ):
                #     search_docs = st.toggle(
                #         ui.search_docs_checkbox_label,
                #         value=False,
                #         disabled=False,
                #         key="search_option_disabled",
                #     )
                # else:
                #     search_docs = st.toggle(
                #         ui.search_docs_checkbox_label,
                #         value=False,
                #         disabled=True,
                #         key="search_option_disabled",
                #     )

                # search_knowledge_base = True
                # search_online = st.toggle(ui.search_internet_checkbox_label, value=False)
                # search_wikipedia = False
                # search_arxiv = False
                # # search_docs = False

                # search_docs_option = None

                # st.session_state["chat_disabled"] = False

                # current_top_k_mappings = f"{search_knowledge_base}_{search_online}_{search_wikipedia}_{search_arxiv}_{search_docs_option}"

                # top_k_values = top_k_mappings.get(current_top_k_mappings)

                # # override search_docs_top_k if search_docs_option is isolated
                # if top_k_values is None:
                #     search_knowledge_base_top_k = 0
                #     search_online_top_k = 0
                #     search_wikipedia_top_k = 0
                #     search_arxiv_top_k = 0
                #     search_docs_top_k = 16
                # else:
                #     search_knowledge_base_top_k = top_k_values.get(
                #         "search_knowledge_base_top_k", 0
                #     )
                #     search_online_top_k = top_k_values.get("search_online_top_k", 0)
                #     search_wikipedia_top_k = top_k_values.get(
                #         "search_wikipedia_top_k", 0
                #     )
                #     search_arxiv_top_k = top_k_values.get("search_arxiv_top_k", 0)
                #     search_docs_top_k = top_k_values.get("search_docs_top_k", 0)

            st.markdown("🔥 限时限量免费开放", help="Limited time and quantity free access")
            st.markdown("🏹 如需更佳体验，请前往 [Kaiwu](https://www.kaiwu.info)", help="ChatGPT 4o and chat history archives")

            col_newchat, col_delete = st.columns([1, 1])
            with col_newchat:

                def init_new_chat():
                    keys_to_delete = [
                        "selected_chat_id",
                        "timestamp",
                        "first_run",
                        "messages",
                        "xata_history",
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
                key="selected_chat_id",
            )

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
        try:
            if "chat_disabled" not in st.session_state:
                st.session_state["chat_disabled"] = False

            if "xata_history_refresh" not in st.session_state:
                user_query = st.chat_input(
                    placeholder=ui.chat_human_placeholder,
                    disabled=st.session_state["chat_disabled"],
                )
                if user_query:
                    beginDatetime = get_begin_datetime()
                    if (
                        "count_chat_history" not in st.session_state
                        or "begin_hour" not in st.session_state
                    ):
                        st.session_state["begin_hour"] = beginDatetime.hour
                        st.session_state["count_chat_history"] = count_chat_history(
                            st.session_state["username"], beginDatetime
                        )
                    else:
                        if (
                            st.session_state["begin_hour"] != beginDatetime.hour
                            or st.session_state["count_chat_history"] % 10 == 0
                        ):
                            st.session_state["begin_hour"] = beginDatetime.hour
                            st.session_state["count_chat_history"] = count_chat_history(
                                st.session_state["username"], beginDatetime
                            )

                    if (
                        not (
                            "subsription" in st.session_state
                            and st.session_state["subsription"] == "Elite"
                        )
                    ) and st.session_state["count_chat_history"] > 39:
                        time_range_str = (
                            str(beginDatetime.hour)
                            + ":00 - "
                            + str(beginDatetime.hour + 3)
                            + ":00"
                        )
                        st.chat_message("ai", avatar=ui.chat_ai_avatar).markdown(
                            "You have reached the usage limit for this time range (UTC "
                            + time_range_str
                            + "). Please try again later. (您已达到 UTC "
                            + time_range_str
                            + " 时间范围的使用限制，请稍后再试。)"
                        )

                    else:
                        st.chat_message("human", avatar=ui.chat_user_avatar).markdown(
                            user_query
                        )
                        st.session_state["messages"].append(
                            {"role": "human", "content": user_query}
                        )
                        human_message = HumanMessage(
                            content=user_query,
                            additional_kwargs={"id": st.session_state["username"]},
                        )
                        st.session_state["xata_history"].add_message(human_message)

                        # check text sensitivity
                        answer = check_text_sensitivity(user_query)["answer"]
                        if answer is not None:
                            with st.chat_message("ai", avatar=ui.chat_ai_avatar):
                                st.markdown(answer)
                                st.session_state["messages"].append(
                                    {
                                        "role": "ai",
                                        "content": answer,
                                    }
                                )
                                ai_message = AIMessage(
                                    content=answer,
                                    additional_kwargs={
                                        "id": st.session_state["username"]
                                    },
                                )
                                st.session_state["xata_history"].add_message(ai_message)
                                st.session_state["count_chat_history"] += 1
                        else:
                            current_message = st.session_state["messages"][-8:][1:][:-1]
                            for item in current_message:
                                item.pop("avatar", None)

                            chat_history_recent = str(current_message)

                            if (
                                search_sci
                                or search_online
                                or search_report
                                or search_patent
                                or search_standard
                            ):
                                formatted_messages = str(
                                    [
                                        (msg["role"], msg["content"])
                                        for msg in st.session_state["messages"][1:]
                                    ]
                                )

                                func_calling_response = func_calling_chain(
                                    api_key, llm_model, openai_api_base
                                ).invoke({"input": formatted_messages})

                                query = func_calling_response.get("query")

                                # try:
                                #     created_at = json.loads(
                                #         func_calling_response.get("created_at", None)
                                #     )
                                # except TypeError:
                                #     created_at = None

                                # source = func_calling_response.get("source", None)

                                # filters = {}
                                # if created_at:
                                #     filters["created_at"] = created_at
                                # if source:
                                #     filters["source"] = source

                                # docs_response = []
                                # docs_response.extend(
                                #     search_sci_service(
                                #         query=query,
                                #         filters=filters,
                                #         top_k=3,
                                #     )
                                # )
                                # docs_response.extend(
                                #     search_internet(query, top_k=3)
                                # )
                                docs_response = asyncio.run(
                                    concurrent_search_service(
                                        urls=search_list, query=query
                                    )
                                )

                                input = f"""Must Follow:
    - Respond to "{user_query}" by using information from "{docs_response}" (if available) and your own knowledge to provide a logical, clear, and critically analyzed reply in the same language.
    - Use the chat context from "{chat_history_recent}" (if available) to adjust the level of detail in your response.
    - Employ bullet points selectively, where they add clarity or organization.
    - Cite sources in main text using the Author-Date citation style where applicable.
    - Provide a list of references in markdown format of [title.journal.authors.date.](hyperlinks) at the end (or just the source file name), only for the references mentioned in the generated text.
    - Use LaTeX quoted by '$' or '$$' within markdown to render mathematical formulas.

    Must Avoid:
    - Repeat the human's query.
    - Translate cited references into the query's language.
    - Preface responses with any designation such as "AI:"."""

                            else:
                                input = f"""Respond to "{user_query}". If "{chat_history_recent}" is not empty, use it as chat context."""

                            with st.chat_message("ai", avatar=ui.chat_ai_avatar):
                                st_callback = StreamHandler(st.empty())
                                response = main_chain(
                                    api_key, llm_model, openai_api_base
                                ).invoke(
                                    {"input": input},
                                    {"callbacks": [st_callback]},
                                )

                                st.session_state["messages"].append(
                                    {
                                        "role": "ai",
                                        "content": response,
                                    }
                                )
                                ai_message = AIMessage(
                                    content=response,
                                    additional_kwargs={
                                        "id": st.session_state["username"]
                                    },
                                )
                                st.session_state["xata_history"].add_message(ai_message)
                                st.session_state["count_chat_history"] += 1

                        if len(st.session_state["messages"]) == 3:
                            st.session_state["xata_history_refresh"] = True
                            st.rerun()
            else:
                user_query = st.chat_input(
                    placeholder=ui.chat_human_placeholder,
                    disabled=st.session_state["chat_disabled"],
                )
                del st.session_state["xata_history_refresh"]

        except Exception as e:
            st.error(e)
            # st.error(ui.chat_error_message)

    if __name__ == "__main__":
        main()
