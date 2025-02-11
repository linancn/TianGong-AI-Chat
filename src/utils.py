"""utilities used in the app"""

import asyncio
import os
import random
import re
import string
import time
from datetime import datetime

import aiohttp
import pytz
import streamlit as st
import weaviate
from collections import defaultdict
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery, Filter


os.environ["LANGCHAIN_VERBOSE"] = str(st.secrets["langchain_verbose"])
os.environ["PASSWORD"] = st.secrets["password"]
os.environ["X_REGION"] = st.secrets["x_region"]
os.environ["EMAIL"] = st.secrets["email"]
os.environ["PW"] = st.secrets["pw"]
os.environ["REMOTE_BEARER_TOKEN"] = st.secrets["bearer_token"]
os.environ["END_POINT"] = st.secrets["end_point"]

from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama

import ui_config

ui = ui_config.create_ui_from_config()

langchain_verbose = bool(os.environ.get("LANGCHAIN_VERBOSE", "True") == "True")


def random_email(domain="example.com"):
    """
    Generates a random email address in the form of 'username@example.com'.

    :param domain: The domain part of the email address. Defaults to 'example.com'.
    :type domain: str
    :return: A randomly generated email address.
    :rtype: str

    Function Behavior:
        - This function generates a random email address with a random username. The username is composed of lowercase ASCII letters and digits.
    """
    # username length is 5 to 10
    username_length = random.randint(5, 10)
    username = "".join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(username_length)
    )

    return f"{username}@{domain}"


def check_password():
    """
    Validates a user-entered password against an environment variable in a Streamlit application.

    :returns: True if the entered password is correct, False otherwise.
    :rtype: bool

    Function Behavior:
        - Displays a password input field and validates the user's input.
        - Utilizes Streamlit's session state to keep track of password validity across reruns.

    Local Functions:
        - password_entered(): Compares the user-entered password with the stored password in the environment variable.

    Exceptions:
        - Relies on the 'os' library to fetch the stored password, so issues in environment variable could lead to exceptions.

    Note:
        - The "PASSWORD" environment variable must be set for password validation.
        - Deletes the entered password from the session state after validation.
    Security:
        - Ensure that the "PASSWORD" environment variable is securely set to avoid unauthorized access.
    """

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == os.environ["PASSWORD"]:
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


def func_calling_chain():
    """
    Creates and returns a function calling chain for extracting query and filter information from a chat history.

    :returns: An object representing the function calling chain configured to generate structured output based on the provided JSON schema and chat prompt template.
    :rtype: object

    Function Behavior:
        - Defines a JSON schema for structured output that includes query information and date filters.
        - Creates a chat prompt template to instruct the underlying language model on how to generate the desired structured output.
        - Utilizes a language model for structured output generation.
        - Creates the function calling chain with 'create_structured_output_runnable', passing the JSON schema, language model, and chat prompt template as arguments.

    Exceptions:
        - This function depends on external modules and classes like 'SystemMessage', 'HumanMessage', 'ChatPromptTemplate', etc. Exceptions may arise if these dependencies encounter issues.

    Note:
        - It uses a specific language model identified by 'llm_model' for structured output generation. Ensure that 'llm_model' is properly initialized and available for use to avoid unexpected issues.
    """
    func_calling_json_schema = {
        "title": "get_querys_and_filters_to_search_database",
        "description": "Extract the queries and filters for database searching",
        "type": "object",
        "properties": {
            "next_query": {
                "title": "Query",
                "description": "The next query extracted for a vector database semantic search from a chat history. Translate the query into accurate English if it is not already in English.",
                "type": "string",
            },
        },
        "required": ["next_query"],
    }

    prompt_func_calling_msgs = [
        HumanMessage(
            content="You are a world-class algorithm for extracting the next query and filters for searching from a chat history. Make sure to answer in the correct structured format."
        ),
        HumanMessagePromptTemplate.from_template("The chat history:\n{input}"),
    ]

    prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

    # llm_func_calling = ChatOpenAI(model_name=llm_model, temperature=0, streaming=False)
    # llm_func_calling = ChatOpenAI(
    #     api_key=api_key,
    #     model_name=llm_model,
    #     temperature=0.1,
    #     streaming=False,
    #     openai_api_base=openai_api_base,
    # )

    llm_func_calling = ChatOllama(
        model=st.secrets["base_model"],
        disable_streaming=True,
        verbose=langchain_verbose,
    )

    func_calling_chain = prompt_func_calling | llm_func_calling.with_structured_output(
        func_calling_json_schema
    )

    return func_calling_chain


async def fetch(session, url, query, results_per_url, headers):
    async with session.post(
        os.environ["END_POINT"] + url,
        headers=headers,
        json={"query": query, "topK": results_per_url},
    ) as response:
        if response.status == 200:
            try:
                return await response.json()
            except aiohttp.ContentTypeError:
                return {"error": "Invalid JSON response"}
        else:
            return {"error": f"Request failed with status code {response.status}"}


async def concurrent_search_service(urls: list, query: str, top_k: int = 8):
    """
    Perform concurrent search requests to multiple URLs with specified query and filters.

    Args:
        urls (list): List of endpoint URLs to send the requests to.
        query (str): The search query string.
        top_k (int): The maximum number of results to retrieve per URL.

    Returns:
        list: A list of responses from all the URLs.
    """
    num_urls = len(urls)
    results_per_url = max(1, min(8, top_k // num_urls))

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['REMOTE_BEARER_TOKEN']}",
        "email": os.environ["EMAIL"],
        "password": os.environ["PW"],
        "x-region": os.environ["X_REGION"],
    }

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url, query, results_per_url, headers) for url in urls]
        return await asyncio.gather(*tasks)


def main_chain():
    """
    Creates and returns a main Large Language Model (LLM) chain configured to produce responses only to science-related queries while avoiding sensitive topics.

    :return: A configured LLM chain object for producing responses that adhere to the defined conditions.
    :rtype: Object

    Function Behavior:
        - Initializes a ChatOpenAI instance for a specific language model with streaming enabled.
        - Configures a prompt template instructing the model to strictly respond to science-related questions while avoiding sensitive topics.
        - Constructs and returns an LLMChain instance, which uses the configured language model and prompt template.

    Exceptions:
        - Exceptions could propagate from underlying dependencies like the ChatOpenAI or LLMChain classes.
        - TypeError could be raised if internal configurations within the function do not match the expected types.
    """

    llm_chat = ChatOllama(
        model=st.secrets["reasoning_model"],
        disable_streaming=False,
        verbose=langchain_verbose,
    )

    template = """{input}"""

    prompt = PromptTemplate(
        input_variables=["input"],
        template=template,
    )

    chain = prompt | llm_chat | StrOutputParser()

    return chain


class ThinkStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""
        self.think_content = ""
        self.after_think_content = ""
        self.found_think_end = False  # 标记是否已找到 </think>

        with st.expander("思维链...", expanded=True, icon="🤔"):
            self.think_container = st.empty()

        self.after_think_container = st.empty()

        self.start_marker = "<think>"
        self.end_marker = "</think>"

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token

        if not self.found_think_end:
            # 还没找到 </think>，先按原逻辑截取 <think> 内容
            start_idx = self.text.find(self.start_marker)
            if start_idx != -1:
                start_idx += len(self.start_marker)
                think_part = self.text[start_idx:]
                end_idx = think_part.find(self.end_marker)
                if end_idx == -1:
                    self.think_content = think_part
                else:
                    self.think_content = think_part[:end_idx]
                    # 标记已经找到 </think>
                    absolute_end_idx = start_idx + end_idx
                    self.found_think_end = True
                    # 初始化 after_think_content
                    self.after_think_content = self.text[
                        absolute_end_idx + len(self.end_marker) :
                    ]

                # 更新 <think> 容器
                self.think_container.markdown(self.think_content)

                # 如果这时刚刚找到 </think>，也要更新 after_think_container
                if self.found_think_end:
                    self.after_think_container.markdown(self.after_think_content)

        else:
            # 如果已经找到 </think>，就持续更新后续文本
            # after_think_content = self.text[上次提取后的位置:] ...
            # 这里可以直接取 self.after_think_content = self.text[??? :]
            # 也可以维护一个 index 变量，或每次更新新的增量
            self.after_think_content = self.text.split(self.end_marker, 1)[-1]
            self.after_think_container.markdown(self.after_think_content)


def xata_chat_history():
    """
    Creates and returns an instance of XataChatMessageHistory to manage chat history based on the provided session ID.

    :param _session_id: The session ID for which chat history needs to be managed.
    :type _session_id: str
    :return: An instance of XataChatMessageHistory configured with the session ID, API key, database URL, and table name.
    :rtype: XataChatMessageHistory object

    Function Behavior:
        - Initializes a XataChatMessageHistory instance using the given session ID, API key from the environment, database URL from the environment, and a predefined table name.
        - Returns the initialized instance for managing the chat history related to the session.

    Exceptions:
        - KeyError could be raised if the required environment variables ("XATA_API_KEY" or "XATA_DATABASE_URL") are not set.
        - Exceptions could propagate from the XataChatMessageHistory class if initialization fails.
    """

    chat_history = StreamlitChatMessageHistory(key="chat_messages")

    return chat_history


# decorator
def enable_chat_history(func):
    """
    A decorator to enable chat history functionality in the Streamlit application.

    :param func: The function to be wrapped by this decorator.
    :type func: Callable
    :return: The wrapped function with chat history functionality enabled.
    :rtype: Callable

    Function Behavior:
        - Checks if the "xata_history" key is in the Streamlit session state. If not, initializes XataChatMessageHistory with a new session ID and stores it in the session state.
        - Checks if the "messages" key is in the Streamlit session state. If not, initializes it with the assistant's welcome message.
        - Iterates through the stored messages and displays them in the Streamlit UI.
        - Executes the original function passed to the decorator.

    Usage:
        @enable_chat_history
        def your_function():
            # Your code here
    """

    if "xata_history" not in st.session_state:
        st.session_state["xata_history"] = xata_chat_history()
    # to show chat history on ui
    if "messages" not in st.session_state or len(st.session_state["messages"]) == 1:

        welcome_message_text = ui.chat_ai_welcome.format(
            username="there", subscription="free"
        )

        st.session_state["messages"] = [
            {
                "role": "ai",
                "avatar": ui.chat_ai_avatar,
                "content": welcome_message_text,
            }
        ]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"], avatar=msg["avatar"]).write(msg["content"])
        # if "avatar" in msg:
        #     st.chat_message(msg["role"], avatar=msg["avatar"]).write(msg["content"])
        # else:
        #     st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)

    return execute


def is_valid_email(email: str) -> bool:
    """
    Check if the given string is a valid email address.

    Args:
    - email (str): String to check.

    Returns:
    - bool: True if valid email, False otherwise.
    """
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))


def convert_history_to_message(history):
    """
    Converts a chat history object into a dictionary containing the role and content of the message.

    :param history: The chat history object to convert.
    :type history: list
    :returns: A dictionary containing the 'role' and 'content' of the message. If it's an AIMessage, an additional 'avatar' field is included.
    :rtype: dict

    Function Behavior:
        - Checks the type of the incoming history object.
        - Transforms it into a dictionary containing the role ('human' or 'ai') and the content of the message.
    """
    if isinstance(history, HumanMessage):
        return {
            "role": "human",
            "avatar": ui.chat_user_avatar,
            "content": history.content,
        }
    elif isinstance(history, AIMessage):
        return {
            "role": "ai",
            "avatar": ui.chat_ai_avatar,
            "content": history.content,
        }


def initialize_messages(history):
    """
    Initializes a list of chat messages based on the given chat history.

    :param history: The list of chat history objects to initialize the messages from.
    :type history: list
    :returns: A list of dictionaries containing the 'role', 'content', and optionally 'avatar' of each message, with a welcome message inserted at the beginning.
    :rtype: list of dicts

    Function Behavior:
        - Converts each message in the chat history to a dictionary format using the `convert_history_to_message` function.
        - Inserts a welcome message at the beginning of the list.

    Exceptions:
        - Exceptions that may propagate from the `convert_history_to_message` function.
    """
    # convert history to message
    messages = [convert_history_to_message(message) for message in history]

    welcome_message_text = ui.chat_ai_welcome.format(
        username="there", subscription="free"
    )

    # add welcome message
    welcome_message = {
        "role": "ai",
        "avatar": ui.chat_ai_avatar,
        "content": welcome_message_text,
    }
    messages.insert(0, welcome_message)

    return messages


def get_begin_datetime():
    now = datetime.now(pytz.UTC)
    beginHour = (now.hour // 3) * 3
    return datetime(now.year, now.month, now.day, beginHour)


client = weaviate.connect_to_custom(
    http_host=st.secrets["weaviate_http_host"],  # Hostname for the HTTP API connection
    http_port=st.secrets["weaviate_http_port"],  # Default is 80, WCD uses 443
    http_secure=False,  # Whether to use https (secure) for the HTTP API connection
    grpc_host=st.secrets["weaviate_grpc_host"],  # Hostname for the gRPC API connection
    grpc_port=st.secrets["weaviate_grpc_port"],  # Default is 50051, WCD uses 443
    grpc_secure=False,  # Whether to use a secure channel for the gRPC API connection
    auth_credentials=Auth.api_key(
        st.secrets["weaviate_api_key"]
    ),  # API key for authentication
)
collection = client.collections.get("tiangong")


def weaviate_hybrid_search(query: str, top_k: int = 8):
    """
    Performs a similarity search on Weaviate's vector database based on a given query and returns a list of relevant documents.

    :param query: The query to be used for similarity search in Weaviate's vector database.
    :type query: str
    :param top_k: The number of top matching documents to return. Defaults to 16.
    :type top_k: int or None
    :returns: A list of dictionaries, each containing the content and source of the matched documents. The function returns an empty list if 'top_k' is set to 0.
    :rtype: list of dicts

    Function Behavior:
        - Initializes Weaviate with the specified API key and environment.
        - Conducts a similarity search based on the provided query.
        - Extracts and formats the relevant document information before returning.

    Exceptions:
        - This function relies on Weaviate and Python's os library. Exceptions could propagate if there are issues related to API keys, environment variables, or Weaviate initialization.
        - TypeError could be raised if the types of 'query' or 'top_k' do not match the expected types.

    Note:
        - Ensure the Weaviate API key and environment variables are set before running this function.
        - The function uses 'OpenAIEmbeddings' to initialize Weaviate's vector store, which should be compatible with the embeddings in the Weaviate index.
    """

    if top_k == 0:
        return []

    hybrid_response = collection.query.hybrid(
        query=query,
        target_vector="content",
        query_properties=["content"],
        alpha=0.3,
        return_metadata=MetadataQuery(score=True, explain_score=True),
        limit=top_k,
    )

    docs_list = []
    for doc in hybrid_response.objects:
        docs_list.append(
            {"content": doc.properties["content"], "source": doc.properties["source"]}
        )

    client.close()

    return docs_list


def weaviate_hybrid_search_extention(query, top_k: int = 8, ext_k: int = 1):
    hybrid_search_results = collection.query.hybrid(
        query=query,
        target_vector="content",
        query_properties=["content"],
        alpha=0.3,
        return_metadata=MetadataQuery(score=True, explain_score=True),
        limit=top_k,
    )

    original_search_results = hybrid_search_results.objects

    doc_chunks = defaultdict(list)
    doc_sources = {}
    added_chunks = set()

    for result in original_search_results:
        properties = result.properties
        content = properties["content"]
        doc_chunk_id = properties["doc_chunk_id"]
        doc_uuid, chunk_id_str = doc_chunk_id.split("_")
        chunk_id = int(chunk_id_str)

        if doc_uuid not in doc_sources and "source" in properties:
            doc_sources[doc_uuid] = properties["source"]

        if (doc_uuid, chunk_id) not in added_chunks:
            doc_chunks[doc_uuid].append((chunk_id, content))
            added_chunks.add((doc_uuid, chunk_id))

        # Extend backward and forward using ext_k
        for i in range(1, ext_k + 1):
            # Fetch previous chunk
            target_chunk_before = chunk_id - i
            if (
                target_chunk_before >= 0
                and (doc_uuid, target_chunk_before) not in added_chunks
            ):
                before_response = collection.query.fetch_objects(
                    filters=Filter.by_property("doc_chunk_id").equal(
                        f"{doc_uuid}_{target_chunk_before}"
                    ),
                )
                if before_response.objects:
                    before_obj = before_response.objects[0]
                    before_content = before_obj.properties["content"]
                    if (
                        doc_uuid not in doc_sources
                        and "source" in before_obj.properties
                    ):
                        doc_sources[doc_uuid] = before_obj.properties["source"]
                    doc_chunks[doc_uuid].append((target_chunk_before, before_content))
                    added_chunks.add((doc_uuid, target_chunk_before))

            # Fetch following chunk
            total_chunk_count = collection.aggregate.over_all(
                total_count=True,
                filters=Filter.by_property("doc_chunk_id").like(f"{doc_uuid}*"),
            ).total_count
            target_chunk_after = chunk_id + i
            if (
                target_chunk_after <= total_chunk_count
                and (doc_uuid, target_chunk_after) not in added_chunks
            ):
                after_response = collection.query.fetch_objects(
                    filters=Filter.by_property("doc_chunk_id").equal(
                        f"{doc_uuid}_{target_chunk_after}"
                    ),
                )
                if after_response.objects:
                    after_obj = after_response.objects[0]
                    after_content = after_obj.properties["content"]
                    if doc_uuid not in doc_sources and "source" in after_obj.properties:
                        doc_sources[doc_uuid] = after_obj.properties["source"]
                    doc_chunks[doc_uuid].append((target_chunk_after, after_content))
                    added_chunks.add((doc_uuid, target_chunk_after))

    for doc_uuid in doc_chunks:
        doc_chunks[doc_uuid].sort(key=lambda x: x[0])

    docs_list = []
    for doc_uuid, chunks in doc_chunks.items():
        combined_content = "".join(chunk_content for _, chunk_content in chunks)
        source = doc_sources.get(doc_uuid, "")
        docs_list.append({"content": combined_content, "source": source})

    client.close()
    return docs_list
