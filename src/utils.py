"""utilities used in the app"""

import os
import random
import re
import string
import time
from datetime import datetime

import pytz
import streamlit as st
import aiohttp
import asyncio

os.environ["XATA_API_KEY"] = st.secrets["xata_api_key"]
os.environ["XATA_DATABASE_URL"] = st.secrets["xata_db_url"]
os.environ["LANGCHAIN_VERBOSE"] = str(st.secrets["langchain_verbose"])
os.environ["PASSWORD"] = st.secrets["password"]
os.environ["X_REGION"] = st.secrets["x_region"]
os.environ["EMAIL"] = st.secrets["email"]
os.environ["PW"] = st.secrets["pw"]
os.environ["REMOTE_BEARER_TOKEN"] = st.secrets["bearer_token"]
os.environ["END_POINT"] = st.secrets["end_point"]

os.environ["QIANFAN_AK"] = st.secrets["qianfan_ak"]
os.environ["QIANFAN_SK"] = st.secrets["qianfan_sk"]


from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_message_histories import XataChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
# from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint
from langchain_deepseek import ChatDeepSeek
from xata.client import XataClient

# from aip import AipSpeech
# import ffmpeg

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


# def convert_audio_in_memory(input_bytes):
    # try:
    #     input_stream = ffmpeg.input('pipe:0', format='webm')  # 或 format='matroska'
    #     output_stream = ffmpeg.output(
    #         input_stream,
    #         'pipe:1',
    #         format='wav',
    #         acodec='pcm_s16le',
    #         ar='16000',
    #         ac='1'
    #     )

    #     process = ffmpeg.run_async(
    #         output_stream,
    #         pipe_stdin=True,
    #         pipe_stdout=True,
    #         pipe_stderr=True
    #     )

    #     stdout, stderr = process.communicate(input=input_bytes)

    #     if process.returncode != 0:
    #         raise ffmpeg.Error('FFmpeg转换失败', stdout, stderr)

    #     return stdout

    # except ffmpeg.Error as e:
    #     print('转换失败！错误信息：')
    #     print(e.stderr.decode())
    #     return None

# def voice_to_text(audio_bytes):
#     audio_bytes = convert_audio_in_memory(audio_bytes)
#     client = AipSpeech(st.secrets["voice_app_id"], st.secrets["voice_app_key"], st.secrets["voice_app_secret"])
#     text = client.asr(audio_bytes, "wav", 16000, {"dev_pid": 1537})
#     return text


def func_calling_chain(api_key, llm_model, openai_api_base):
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
            # "source": {
            #     "title": "Source Filter",
            #     "description": "Journal Name or Source extracted for a vector database semantic search, MUST be in upper case.",
            #     "type": "string",
            #     "enum": [
            #         "AGRICULTURE, ECOSYSTEMS & ENVIRONMENT",
            #         "ANNUAL REVIEW OF ECOLOGY, EVOLUTION, AND SYSTEMATICS",
            #         "ANNUAL REVIEW OF ENVIRONMENT AND RESOURCES",
            #         "APPLIED CATALYSIS B: ENVIRONMENTAL",
            #         "BIOGEOSCIENCES",
            #         "BIOLOGICAL CONSERVATION",
            #         "BIOTECHNOLOGY ADVANCES",
            #         "CONSERVATION BIOLOGY",
            #         "CONSERVATION LETTERS",
            #         "CRITICAL REVIEWS IN ENVIRONMENTAL SCIENCE AND TECHNOLOGY",
            #         "DIVERSITY AND DISTRIBUTIONS",
            #         "ECOGRAPHY",
            #         "ECOLOGICAL APPLICATIONS",
            #         "ECOLOGICAL ECONOMICS",
            #         "ECOLOGICAL MONOGRAPHS",
            #         "ECOLOGY",
            #         "ECOLOGY LETTERS",
            #         "ECONOMIC SYSTEMS RESEARCH",
            #         "ECOSYSTEM HEALTH AND SUSTAINABILITY",
            #         "ECOSYSTEM SERVICES",
            #         "ECOSYSTEMS",
            #         "ENERGY & ENVIRONMENTAL SCIENCE",
            #         "ENVIRONMENT INTERNATIONAL",
            #         "ENVIRONMENTAL CHEMISTRY LETTERS",
            #         "ENVIRONMENTAL HEALTH PERSPECTIVES",
            #         "ENVIRONMENTAL POLLUTION",
            #         "ENVIRONMENTAL SCIENCE & TECHNOLOGY",
            #         "ENVIRONMENTAL SCIENCE & TECHNOLOGY LETTERS",
            #         "ENVIRONMENTAL SCIENCE AND ECOTECHNOLOGY",
            #         "ENVIRONMENTAL SCIENCE AND POLLUTION RESEARCH",
            #         "EVOLUTION",
            #         "FOREST ECOSYSTEMS",
            #         "FRONTIERS IN ECOLOGY AND THE ENVIRONMENT",
            #         "FRONTIERS OF ENVIRONMENTAL SCIENCE & ENGINEERING",
            #         "FUNCTIONAL ECOLOGY",
            #         "GLOBAL CHANGE BIOLOGY",
            #         "GLOBAL ECOLOGY AND BIOGEOGRAPHY",
            #         "GLOBAL ENVIRONMENTAL CHANGE",
            #         "INTERNATIONAL SOIL AND WATER CONSERVATION RESEARCH",
            #         "JOURNAL OF ANIMAL ECOLOGY",
            #         "JOURNAL OF APPLIED ECOLOGY",
            #         "JOURNAL OF BIOGEOGRAPHY",
            #         "JOURNAL OF CLEANER PRODUCTION",
            #         "JOURNAL OF ECOLOGY",
            #         "JOURNAL OF ENVIRONMENTAL INFORMATICS",
            #         "JOURNAL OF ENVIRONMENTAL MANAGEMENT",
            #         "JOURNAL OF HAZARDOUS MATERIALS",
            #         "JOURNAL OF INDUSTRIAL ECOLOGY",
            #         "JOURNAL OF PLANT ECOLOGY",
            #         "LANDSCAPE AND URBAN PLANNING",
            #         "LANDSCAPE ECOLOGY",
            #         "METHODS IN ECOLOGY AND EVOLUTION",
            #         "MICROBIOME",
            #         "MOLECULAR ECOLOGY",
            #         "NATURE",
            #         "NATURE CLIMATE CHANGE",
            #         "NATURE COMMUNICATIONS",
            #         "NATURE ECOLOGY & EVOLUTION",
            #         "NATURE ENERGY",
            #         "NATURE REVIEWS EARTH & ENVIRONMENT",
            #         "NATURE SUSTAINABILITY",
            #         "ONE EARTH",
            #         "PEOPLE AND NATURE",
            #         "PROCEEDINGS OF THE NATIONAL ACADEMY OF SCIENCES",
            #         "PROCEEDINGS OF THE ROYAL SOCIETY B: BIOLOGICAL SCIENCES",
            #         "RENEWABLE AND SUSTAINABLE ENERGY REVIEWS",
            #         "RESOURCES, CONSERVATION AND RECYCLING",
            #         "REVIEWS IN ENVIRONMENTAL SCIENCE AND BIO/TECHNOLOGY",
            #         "SCIENCE",
            #         "SCIENCE ADVANCES",
            #         "SCIENCE OF THE TOTAL ENVIRONMENT",
            #         "SCIENTIFIC DATA",
            #         "SUSTAINABLE CITIES AND SOCIETY",
            #         "SUSTAINABLE MATERIALS AND TECHNOLOGIES",
            #         "SUSTAINABLE PRODUCTION AND CONSUMPTION",
            #         "THE AMERICAN NATURALIST",
            #         "THE INTERNATIONAL JOURNAL OF LIFE CYCLE ASSESSMENT",
            #         "THE ISME JOURNAL",
            #         "THE LANCET PLANETARY HEALTH",
            #         "TRENDS IN ECOLOGY & EVOLUTION",
            #         "WASTE MANAGEMENT",
            #         "WATER RESEARCH",
            #     ],
            # },
            # "created_at": {
            #     "title": "Date Filter",
            #     "description": 'Date extracted for a vector database semantic search, in MongoDB\'s query and projection operators, in format like {"$gte": 1609459200.0, "$lte": 1640908800.0}',
            #     "type": "string",
            # },
        },
        "required": ["next_query"],
    }

    prompt_func_calling_msgs = [
        SystemMessage(
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

    llm_func_calling = ChatOpenAI(
        api_key=st.secrets["openai_api_key_zhipu"],
        model_name=st.secrets["llm_model_zhipu"],
        temperature=0.1,
        streaming=False,
        openai_api_base=st.secrets["openai_api_base_zhipu"],
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


def main_chain(api_key, llm_model, openai_api_base, baidu_llm):
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

    # if baidu_llm:
    #     llm_chat = QianfanLLMEndpoint(
    #         endpoint="ernie-4.0-turbo-128k",
    #         model="ERNIE-4.0-Turbo-8K",
    #         temperature=0.1,
    #         streaming=True,
    #     )
    # else:
    #     llm_chat = ChatOpenAI(
    #         api_key=api_key,
    #         model_name=llm_model,
    #         temperature=0.1,
    #         streaming=True,
    #         verbose=langchain_verbose,
    #         openai_api_base=openai_api_base,
    #     )

    # llm_chat = ChatOpenAI(
    #     api_key=st.secrets["openai_api_key_baidu"],
    #     model_name=st.secrets["llm_model_baidu"],
    #     temperature=0.1,
    #     streaming=True,
    #     verbose=langchain_verbose,
    #     openai_api_base=st.secrets["openai_api_base_baidu"],
    # )

    # llm_chat = ChatOpenAI(
    #     api_key=st.secrets["openai_api_key_ds"],
    #     model_name=st.secrets["llm_model_ds"],
    #     temperature=0.1,
    #     streaming=True,
    #     verbose=langchain_verbose,
    #     openai_api_base=st.secrets["openai_api_base_ds"],
    # )

    llm_chat = ChatDeepSeek(
    model=st.secrets["llm_model_ds"],
    api_key=st.secrets["openai_api_key_ds"],
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
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

        with st.expander("思考中...", expanded=True, icon="🤔"):
            self.think_container = st.empty()

        self.after_think_container = st.empty()

        self.start_marker = "<think>"
        self.end_marker = "</think>"

    def on_llm_new_token(self, token: str, chunk, **kwargs) -> None:
        # self.text += token
        self.reasoning_content = chunk.message.additional_kwargs.get("reasoning_content", None)
        # self.text += chunk.message.additional_kwargs.get("reasoning_content", None)
        if self.reasoning_content is not None:
            self.text += self.reasoning_content
            # 更新 <think> 容器
            self.think_container.markdown(self.text)
        else:
            self.after_think_content += chunk.message.content
            self.after_think_container.markdown(self.after_think_content)

class StreamHandler(BaseCallbackHandler):
    """
    A handler class for streaming text to a Streamlit container during the Language Learning Model (LLM) operation.
    """

    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        """
        Callback function for when a new token is generated by the LLM.
        :param token: The newly generated token.
        :type token: str
        :param kwargs: Additional keyword arguments, if any.
        """
        self.text += token
        self.container.markdown(self.text)


def xata_chat_history(_session_id: str):
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

    chat_history = XataChatMessageHistory(
        session_id=_session_id,
        api_key=os.environ["XATA_API_KEY"],
        db_url=os.environ["XATA_DATABASE_URL"],
        table_name="tiangong_memory",
    )

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
        st.session_state["xata_history"] = xata_chat_history(
            _session_id=str(time.time())
        )
    # to show chat history on ui
    if "messages" not in st.session_state or len(st.session_state["messages"]) == 1:
        if "subscription" in st.session_state:
            welcome_message_text = ui.chat_ai_welcome.format(
                username=st.session_state["username"].split("@")[0],
                subscription=st.session_state["subsription"],
            )
        else:
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


def fetch_chat_history(username: str):
    """
    Fetches the chat history from the Xata database, organizing it into a structured format for further use.

    :param username: The username to filter chat history by.
    :type username: str
    :returns: A dictionary where each session ID is mapped to its corresponding chat history entry, formatted with date and content.
    :rtype: dict

    Function Behavior:
        - Utilizes the XataClient class to connect to the Xata database.
        - Executes an SQL query to fetch unique session IDs along with their latest content and timestamp.
        - Formats the timestamp to a readable date and time format and appends it along with the content.
        - Returns the organized chat history as a dictionary where the session IDs are the keys and the formatted chat history entries are the values.

    Exceptions:
        - ConnectionError: Could be raised if there are issues connecting to the Xata database.
        - SQL-related exceptions: Could be raised if the query is incorrect or if there are other database-related issues.
        - TypeError: Could be raised if the types of the returned values do not match the expected types.

    Note:
        - The SQL query used in this function assumes that the Xata database schema has specific columns. If the schema changes, the query may need to be updated.
        - The function returns an empty dictionary if no records are found.
    """
    if is_valid_email(username):
        client = XataClient()
        response = client.sql().query(
            f"""SELECT "sessionId", "content"
    FROM (
        SELECT DISTINCT ON ("sessionId") "sessionId", "xata.createdAt", "content"
        FROM "tiangong_memory"
        WHERE "additionalKwargs"->>'id' = '{username}'
        ORDER BY "sessionId" DESC, "xata.createdAt" ASC
    ) AS subquery"""
        )
        records = response["records"]
        for record in records:
            timestamp = float(record["sessionId"])
            record["entry"] = (
                datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                + " : "
                + record["content"]
            )

        table_map = {item["sessionId"]: item["entry"] for item in records}

        return table_map
    else:
        return {}


def delete_chat_history(session_id):
    """
    Deletes the chat history associated with a specific session ID from the Xata database.

    :param session_id: The session ID for which the chat history needs to be deleted.
    :type session_id: str

    Function Behavior:
        - Utilizes the XataClient class to connect to the Xata database.
        - Executes an SQL query to delete all records associated with the given session ID.

    Exceptions:
        - ConnectionError: Could be raised if there are issues connecting to the Xata database.
        - SQL-related exceptions: Could be raised if the query is incorrect or if there are other database-related issues.

    Note:
        - The function does not check whether the session ID exists in the database before attempting the delete operation.
        - Ensure that you want to permanently delete the chat history for the specified session ID before calling this function.
    """

    client = XataClient()
    client.sql().query(
        'DELETE FROM "tiangong_memory" WHERE "sessionId" = $1',
        [session_id],
    )


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

    if "subscription" in st.session_state:
        welcome_message_text = ui.chat_ai_welcome.format(
            username=st.session_state["username"].split("@")[0],
            subscription=st.session_state["subsription"],
        )
    else:
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


def count_chat_history(username: str, beginDatetime: datetime):
    if is_valid_email(username):
        client = XataClient()
        response = client.sql().query(
            f"""SELECT count(*) as c
    FROM "tiangong_memory"
    WHERE "additionalKwargs"->>'id' = '{username}' and "xata.createdAt" > '{beginDatetime.strftime("%Y-%m-%d %H:%M:%S")}' and "type" = 'ai'
    """
        )
        records = response["records"]
        return records[0]["c"]
    else:
        return 0
