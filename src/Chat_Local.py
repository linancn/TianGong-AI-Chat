import streamlit as st
from langchain.schema import AIMessage, HumanMessage

import ui_config
import utils
import wix_oauth as wix_oauth
from sensitivity_checker import check_text_sensitivity
from utils import (
    ThinkStreamHandler,
    weaviate_connection,
    weaviate_hybrid_search_extention,
    func_calling_chain,
    initialize_messages,
    main_chain,
)

ui = ui_config.create_ui_from_config()
st.set_page_config(page_title=ui.page_title, layout="wide", page_icon=ui.page_icon)

# CSS style injection
st.markdown(
    ui.page_markdown,
    unsafe_allow_html=True,
)

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

            if search_sci:
                search_sci_collection = weaviate_connection("tiangong")
            if search_report:
                search_report_collection = weaviate_connection("audit")

        def init_new_chat():
            for key in st.session_state.keys():
                del st.session_state[key]

        new_chat = st.button(
            ui.sidebar_newchat_button_label,
            use_container_width=True,
            on_click=init_new_chat,
        )

        if "first_run" not in st.session_state:
            st.session_state["first_run"] = True
        else:
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
                st.chat_message("human", avatar=ui.chat_user_avatar).markdown(
                    user_query
                )
                st.session_state["messages"].append(
                    {"role": "human", "content": user_query}
                )
                human_message = HumanMessage(
                    content=user_query,
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
                        )
                        st.session_state["xata_history"].add_message(ai_message)
                else:
                    current_message = st.session_state["messages"][-8:][1:][:-1]
                    for item in current_message:
                        item.pop("avatar", None)

                    chat_history_recent = str(current_message)

                    if search_sci or search_report:
                        formatted_messages = str(
                            [
                                (msg["role"], msg["content"])
                                for msg in st.session_state["messages"][1:]
                            ]
                        )

                        func_calling_response = func_calling_chain().invoke(
                            {"input": formatted_messages}
                        )

                        query = func_calling_response.get("query")

                        docs_response = []
                        if search_sci:
                            sci_response = weaviate_hybrid_search_extention(
                                search_sci_collection, query, top_k=8, ext_k=1
                            )
                            docs_response.append(sci_response)

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
                        st_callback = ThinkStreamHandler()
                        response = main_chain().invoke(
                            {"input": input},
                            {"callbacks": [st_callback]},
                        )
                        if "</think>" in response:
                            response = response.split("</think>", 1)[1].strip()

                        st.session_state["messages"].append(
                            {
                                "role": "ai",
                                "content": response,
                            }
                        )
                        ai_message = AIMessage(
                            content=response,
                        )
                        st.session_state["xata_history"].add_message(ai_message)

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
