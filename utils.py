import re
import tempfile
import time
from datetime import datetime

import pinecone
import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import XataChatMessageHistory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.vectorstores import FAISS, Pinecone
from xata.client import XataClient

import ui_config

ui = ui_config.create_ui_from_config()

llm_model = st.secrets["llm_model"]
langchain_verbose = st.secrets["langchain_verbose"]


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


def search_pinecone(query, created_at, top_k=16):
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
        date = datetime.fromtimestamp(doc.metadata["created_at"])
        formatted_date = date.strftime("%Y-%m")  # Format date as 'YYYY-MM'
        source_entry = "[{}. {}. {}.]({})".format(
            doc.metadata["source_id"],
            doc.metadata["author"],
            formatted_date,
            doc.metadata["url"],
        )
        docs_list.append({"content": doc.page_content, "source": source_entry})

    return docs_list


def search_internet(query):
    search = DuckDuckGoSearchResults()
    results = search.run(query)

    pattern = r"\[snippet: (.*?), title: (.*?), link: (.*?)\]"
    matches = re.findall(pattern, results)

    docs = [
        {"snippet": match[0], "title": match[1], "link": match[2]} for match in matches
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


def search_wiki(query):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    response_wiki = wikipedia.run(query)

    return response_wiki


def seach_docs(query, top_k=16):
    docs = st.session_state["faiss_db"].similarity_search(query, k=top_k)
    docs_list = []
    for doc in docs:
        source_entry = doc.metadata["source"]
        docs_list.append({"content": doc.page_content, "source": source_entry})

    return docs_list


def func_calling_chain():
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

    llm_func_calling = ChatOpenAI(model_name=llm_model, temperature=0, streaming=False)

    func_calling_chain = create_structured_output_chain(
        output_schema=func_calling_json_schema,
        llm=llm_func_calling,
        prompt=prompt_func_calling,
        verbose=langchain_verbose,
    )

    return func_calling_chain


def chat_history_chain():
    llm_chat_history = ChatOpenAI(
        model=llm_model,
        temperature=0,
        streaming=False,
        verbose=langchain_verbose,
    )

    template = """Return highly concise and well-organized chat history from: {input}"""
    prompt = PromptTemplate(
        input_variables=["input"],
        template=template,
    )

    chat_history_chain = LLMChain(
        llm=llm_chat_history,
        prompt=prompt,
        verbose=langchain_verbose,
    )

    return chat_history_chain


def main_chain():
    llm_chat = ChatOpenAI(
        model=llm_model,
        temperature=0,
        streaming=True,
        verbose=langchain_verbose,
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
        verbose=langchain_verbose,
    )

    return chain


def xata_chat_history(_session_id: str):
    chat_history = XataChatMessageHistory(
        session_id=_session_id,
        api_key=st.secrets["xata_api_key"],
        db_url=st.secrets["xata_db_url"],
        table_name="tiangong_memory",
    )

    return chat_history

# decorator
def enable_chat_history(func):
    if "xata_history" not in st.session_state:
        st.session_state["xata_history"] = xata_chat_history(
            _session_id=str(time.time())
        )
    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "avatar": ui.chat_ai_avatar,
                "content": ui.chat_ai_welcome,
            }
        ]
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message(msg["role"]).write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message(msg["role"], avatar=msg["avatar"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)

    return execute


def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)


def get_faiss_db(uploaded_files):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=220, chunk_overlap=20
    )
    chunks = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=True) as fp:
                fp.write(uploaded_file.read())
                loader = UnstructuredFileLoader(file_path=fp.name)
                docs = loader.load()
                full_text = docs[0].page_content

            chunk = text_splitter.create_documents(
                texts=[full_text], metadatas=[{"source": uploaded_file.name}]
            )
            chunks.extend(chunk)
        except:
            pass
    if chunks != []:
        embeddings = OpenAIEmbeddings()
        faiss_db = FAISS.from_documents(chunks, embeddings)
    else:
        st.warning(ui.sidebar_file_uploader_error)
        st.stop()

    return faiss_db


class StreamHandler(BaseCallbackHandler):
    
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


def fetch_chat_history():
    """Fetch the chat history."""
    client = XataClient()
    response = client.sql().query(
        'SELECT "sessionId", "content" FROM (SELECT DISTINCT ON ("sessionId") "sessionId", "xata.createdAt", "content" FROM "tiangong_memory" ORDER BY "sessionId", "xata.createdAt" ASC, "content" ASC) AS subquery ORDER BY "xata.createdAt" DESC'
    )
    records = response["records"]
    for record in records:
        timestamp = float(record["sessionId"])
        record["entry"] = (
            datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            + " - "
            + record["content"]
        )

    table_map = {item["sessionId"]: item["entry"] for item in records}

    return table_map


def delete_chat_history(session_id):
    """Delete the chat history by session_id."""
    client = XataClient()
    client.sql().query(
        'DELETE FROM "tiangong_memory" WHERE "sessionId" = $1',
        [session_id],
    )


def convert_history_to_message(history):
    if isinstance(history, HumanMessage):
        return {"role": "user", "content": history.content}
    elif isinstance(history, AIMessage):
        return {
            "role": "assistant",
            "avatar": ui.chat_ai_avatar,
            "content": history.content,
        }


def initialize_messages(history):
    # 将历史消息转换为消息格式
    messages = [convert_history_to_message(message) for message in history]

    # 在最前面加入欢迎消息
    welcome_message = {
        "role": "assistant",
        "avatar": ui.chat_ai_avatar,
        "content": ui.chat_ai_welcome,
    }
    messages.insert(0, welcome_message)

    return messages
