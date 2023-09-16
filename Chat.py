import datetime
import json
import os
import re

import pinecone
import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.vectorstores import Pinecone


import utils
import ui_config
from streaming import StreamHandler

llm_model = st.secrets["llm_model"]
langchain_verbose = st.secrets["langchain_verbose"]

ui = ui_config.create_ui_from_config()

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

        with st.expander(ui.sidebar_expander_title):
            search_internet = st.checkbox(ui.search_internet_checkbox_label, value=True)
            search_docs = st.checkbox(ui.search_docs_checkbox_label, value=False)

        if search_docs:
            search_docs_option = st.radio(
                label=ui.search_docs_options,
                options=(
                    ui.search_docs_options_isolated,
                    ui.search_docs_options_combined,
                ),
                horizontal=True,
            )
            uploaded_files = st.sidebar.file_uploader(
                ui.sidebar_file_uploader_title,
                accept_multiple_files=True,
                type=None,
            )

            if uploaded_files != [] and uploaded_files != st.session_state.get(
                "uploaded_files"
            ):
                st.session_state["uploaded_files"] = uploaded_files
                with st.spinner(ui.sidebar_file_uploader_spinner):
                    st.session_state["faiss_db"] = utils.get_faiss_db(uploaded_files)

    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

    class Basic:
        def __init__(self):
            self.openai_model = llm_model
            self.verbose = langchain_verbose

        def search_pinecone(self, query, created_at, top_k=16):
            embeddings = OpenAIEmbeddings()
            pinecone.init(
                api_key=st.secrets["pinecone_api_key"],
                environment=st.secrets["pinecone_environment"],
            )
            vectorstore = Pinecone.from_existing_index(
                index_name=st.secrets["pinecone_index"],
                embedding=embeddings,
            )
            if created_at is not None:
                docs = vectorstore.similarity_search(
                    query, k=top_k, filter={"created_at": created_at}
                )
            else:
                docs = vectorstore.similarity_search(query, k=top_k)

            docs_list = []
            for doc in docs:
                date = datetime.datetime.fromtimestamp(doc.metadata["created_at"])
                formatted_date = date.strftime("%Y-%m")  # Format date as 'YYYY-MM'
                source_entry = "[{}. {}. {}.]({})".format(
                    doc.metadata["source_id"],
                    doc.metadata["author"],
                    formatted_date,
                    doc.metadata["url"],
                )
                docs_list.append({"content": doc.page_content, "source": source_entry})

            return docs_list

        def search_internet(self, query):
            search = DuckDuckGoSearchResults()
            results = search.run(query)

            pattern = r"\[snippet: (.*?), title: (.*?), link: (.*?)\]"
            matches = re.findall(pattern, results)

            docs = [
                {"snippet": match[0], "title": match[1], "link": match[2]}
                for match in matches
            ]

            docs_list = []

            for doc in docs:
                docs_list.append(
                    {
                        "content": doc["snippet"],
                        "source": "[{}]({})".format(doc["title"], doc["link"]),
                    }
                )

            return docs_list

        def search_wiki(self, query):
            wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
            response_wiki = wikipedia.run(query)

            return response_wiki

        def seach_docs(self, query, top_k=16):
            docs = st.session_state["faiss_db"].similarity_search(query, k=top_k)
            docs_list = []
            for doc in docs:
                source_entry = doc.metadata["source"]
                docs_list.append({"content": doc.page_content, "source": source_entry})

            return docs_list

        def func_calling_chain(self):
            func_calling_json_schema = {
                "title": "get_querys_and_filters_to_search_embeddings",
                "description": "Extract the next queries and filters for a vector database semantic search from a chat history.",
                "type": "object",
                "properties": {
                    "query": {
                        "title": "Query",
                        "description": "The next queries extracted for a vector database semantic search from a chat history in the format of a JSON object",
                        "type": "string",
                    },
                    "created_at": {
                        "title": "Date Filters",
                        "description": 'Date extracted for a vector database semantic search from a chat history, in MongoDB\'s query and projection operators, in format like {"$gte": 1609459200.0, "$lte": 1640908800.0}',
                        "type": "string",
                    },
                    # "author": {
                    #     "title": "Author Filters",
                    #     "description": 'Author(s) extracted in MongoDB\'s query and projection operators, in format like {"$in": ["John Doe"]}',
                    #     "type": "string",
                    # },
                },
                "required": ["query"],
            }

            prompt_func_calling_msgs = [
                SystemMessage(
                    content="You are a world class algorithm for extracting the next queries and filters for a vector database semantic search from a chat history. Make sure to answer in the correct structured format"
                ),
                HumanMessage(content="The chat history:"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]

            prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

            llm_func_calling = ChatOpenAI(
                model_name=self.openai_model, temperature=0, streaming=False
            )

            func_calling_chain = create_structured_output_chain(
                output_schema=func_calling_json_schema,
                llm=llm_func_calling,
                prompt=prompt_func_calling,
                verbose=self.verbose,
            )

            return func_calling_chain

        def chat_history_chain(self):
            llm_chat_history = ChatOpenAI(
                model=self.openai_model,
                temperature=0,
                streaming=False,
                verbose=self.verbose,
            )

            template = """Return highly concise and well-organized chat history from: {input}"""
            prompt = PromptTemplate(
                input_variables=["input"],
                template=template,
            )

            chat_history_chain = LLMChain(
                llm=llm_chat_history,
                prompt=prompt,
                verbose=self.verbose,
            )

            return chat_history_chain

        def main_chain(self):
            llm_chat = ChatOpenAI(
                model=self.openai_model,
                temperature=0,
                streaming=True,
                verbose=self.verbose,
            )

            template = """You MUST ONLY responese to science-related quests.
            DO NOT return any information on politics, ethnicity, gender, national sovereignty, or other sensitive topics.
            {input}
            Use bullet points if a better expression effect can be achieved."""

            prompt = PromptTemplate(
                input_variables=["input"],
                template=template,
            )

            chain = LLMChain(
                llm=llm_chat,
                prompt=prompt,
                verbose=self.verbose,
            )

            return chain

        @utils.enable_chat_history
        def main(self):
            if "chat_history" not in st.session_state:
                chat_history = ChatMessageHistory()
                st.session_state["chat_history"] = chat_history
            else:
                chat_history = st.session_state["chat_history"]

            user_query = st.chat_input(placeholder=ui.chat_human_placeholder)

            if user_query:
                chat_history.add_user_message(user_query)
                utils.display_msg(user_query, "user")

                if len(chat_history.messages) <= 1:
                    func_calling_response = self.func_calling_chain().run(user_query)

                    query = func_calling_response.get("query")
                    try:
                        created_at = json.loads(
                            func_calling_response.get("created_at", None)
                        )
                    except TypeError:
                        created_at = None

                    if search_docs:
                        if search_docs_option == ui.search_docs_options_isolated:
                            docs_response = self.seach_docs(query, top_k=16)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources."""
                        elif search_docs_option == ui.search_docs_options_combinedd:
                            if search_internet:
                                embedding_results = self.search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = self.seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                internet_results = self.search_internet(query)
                                docs_response.extend(internet_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                            elif not search_internet:
                                embedding_results = self.search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = self.seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                    elif not search_docs:
                        if search_internet:
                            embedding_results = self.search_pinecone(
                                query, created_at, top_k=16
                            )
                            internet_results = self.search_internet(query)
                            embedding_results.extend(internet_results)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                        elif not search_internet:
                            embedding_results = self.search_pinecone(
                                query, created_at, top_k=16
                            )
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                else:
                    chat_history_response = self.chat_history_chain()(
                        {"input": chat_history.messages[-7:]},
                    )
                    chat_history_summary = chat_history_response["text"]

                    func_calling_response = self.func_calling_chain().run(
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
                            docs_response = self.seach_docs(query, top_k=16)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources. Current conversation:"{chat_history_summary}"""
                        elif search_docs_option == ui.search_docs_options_combined:
                            if search_internet:
                                embedding_results = self.search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = self.seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                internet_results = self.search_internet(query)
                                docs_response.extend(internet_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""
                            elif not search_internet:
                                embedding_results = self.search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = self.seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""
                    elif not search_docs:
                        if search_internet:
                            embedding_results = self.search_pinecone(
                                query, created_at, top_k=16
                            )
                            internet_results = self.search_internet(query)
                            embedding_results.extend(internet_results)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""
                        elif not search_internet:
                            embedding_results = self.search_pinecone(
                                query, created_at, top_k=16
                            )
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{user_query}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""

                with st.chat_message("assistant", avatar=ui.chat_ai_avatar):
                    st_cb = StreamHandler(st.empty())
                    response = self.main_chain()(
                        {"input": input},
                        callbacks=[st_cb],
                    )
                    chat_history.add_ai_message(response["text"])
                    st.session_state["chat_history"] = chat_history
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "avatar": ui.chat_ai_avatar,
                            "content": response["text"],
                        }
                    )

    if __name__ == "__main__":
        obj = Basic()
        obj.main()
