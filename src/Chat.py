import asyncio
import datetime
import time
import uuid
from datetime import datetime

import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from streamlit.web.server.websocket_headers import _get_websocket_headers
from streamlit_chat_widget import chat_input_widget
from streamlit_float import *

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
            st.subheader(
                "环境生态领域智能助手",
                help="Environment and Ecology Intelligent Assistant",
            )

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
                options=["BAIDU 百度", "ZHIPU 智谱"],
                horizontal=True,
                # index=1,
                help="ernie-4.0-turbo-128k / glm-4-plus",
                # disabled=True,
            )
            if base_model == "ZHIPU 智谱":
                api_key = st.secrets["openai_api_key_zhipu"]
                llm_model = st.secrets["llm_model_zhipu"]
                openai_api_base = st.secrets["openai_api_base_zhipu"]
                baidu_llm = False
            elif base_model == "BAIDU 百度":
                api_key = st.secrets["openai_api_key_baidu"]
                llm_model = st.secrets["llm_model_baidu"]
                openai_api_base = st.secrets["openai_api_base_baidu"]
                baidu_llm = True

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

            st.markdown(
                "🔥 限时限量免费开放", help="Limited time and quantity free access"
            )
            st.markdown(
                "🏹 如需更佳体验，请前往 [Kaiwu](https://www.kaiwu.info)",
                help="ChatGPT 4o and chat history archives",
            )

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
            float_init()
            footer_container = st.container()

            if "chat_disabled" not in st.session_state:
                st.session_state["chat_disabled"] = False

            if "xata_history_refresh" not in st.session_state:
                # user_query = st.chat_input(
                #     placeholder=ui.chat_human_placeholder,
                #     disabled=st.session_state["chat_disabled"],
                # )
                with footer_container:
                    user_input = chat_input_widget(key="user_input")
                footer_container.float(
                    "display:flex; align-items:center;justify-content:center; flex-direction:column; position:fixed; bottom:5px; margin:0; padding:0; z-index:0;"
                )
                if user_input:
                    if "text" in user_input:
                        user_query = user_input["text"]
                    elif "audioFile" in user_input:
                        audio_bytes = bytes(user_input["audioFile"])
                        # st.audio(audio_bytes, format="audio/wav")
                        voice_result = utils.voice_to_text(audio_bytes)["result"]
                        user_query = " ".join(voice_result)

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
                                st.session_state["count_chat_history"] = (
                                    count_chat_history(
                                        st.session_state["username"], beginDatetime
                                    )
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
                            st.chat_message(
                                "human", avatar=ui.chat_user_avatar
                            ).markdown(user_query)
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
                                    st.session_state["xata_history"].add_message(
                                        ai_message
                                    )
                                    st.session_state["count_chat_history"] += 1
                            else:
                                current_message = st.session_state["messages"][-8:][1:][
                                    :-1
                                ]
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

                                    docs_response = asyncio.run(
                                        concurrent_search_service(
                                            urls=search_list, query=query
                                        )
                                    )

                                    input = f"""必须遵循：
    - 使用“{docs_response}”（如果有）和您自己的知识回应“{user_query}”，以用户相同的语言提供逻辑清晰、经过批判性分析的回复。
    - 如果有“{chat_history_recent}”，请利用聊天上下文调整回复的详细程度。
    - 如果没有提供参考或没有上下文的情况，不要要求用户提供，直接回应用户的问题。
    - 有选择地使用项目符号，以提高清晰度或组织性。
    - 在适用情况下，使用 作者-日期 的引用风格在正文中引用来源。
    - 在末尾以Markdown格式提供一个参考文献列表，格式为[标题.期刊.作者.日期.](链接)（或仅文件名），仅包括文本中提到的参考文献。
    - 在Markdown中使用 '$' 或 '$$' 引用LaTeX以渲染数学公式。

    必须避免：
    - 重复用户的查询。
    - 将引用的参考文献翻译成用户查询的语言。
    - 在回复前加上任何标识，如“AI：”。
    """

                                else:
                                    input = f"""回应“{user_query}”。如果“{chat_history_recent}”不为空，请使用其作为聊天上下文。"""

                                with st.chat_message("ai", avatar=ui.chat_ai_avatar):
                                    st_callback = StreamHandler(st.empty())
                                    response = main_chain(
                                        api_key, llm_model, openai_api_base, baidu_llm
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
                                    st.session_state["xata_history"].add_message(
                                        ai_message
                                    )
                                    st.session_state["count_chat_history"] += 1

                            if len(st.session_state["messages"]) == 3:
                                st.session_state["xata_history_refresh"] = True
                                st.rerun()
            else:
                # user_query = st.chat_input(
                #     placeholder=ui.chat_human_placeholder,
                #     disabled=st.session_state["chat_disabled"],
                # )
                with footer_container:
                    user_input = chat_input_widget()
                footer_container.float(
                    "display:flex; align-items:center;justify-content:center; flex-direction:column; position:fixed; bottom:5px; margin:0; padding:0; z-index:0;"
                )
                del st.session_state["xata_history_refresh"]

        except Exception as e:
            # st.error(e)
            st.error(ui.chat_error_message)

    if __name__ == "__main__":
        main()
