import os
import tempfile
import time
from datetime import datetime

import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import XataChatMessageHistory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from xata.client import XataClient

import ui_config

ui = ui_config.create_ui_from_config()


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
        st.session_state["xata_history"] = xata_chat_history(_session_id=str(time.time()))
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