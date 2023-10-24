"""utilities used in the app"""
import io
import os
import random
import re
import string
import tempfile
import time
from collections import Counter
from datetime import datetime

import arxiv
import pdfplumber
import pinecone
import requests
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["XATA_API_KEY"] = st.secrets["xata_api_key"]
os.environ["XATA_DATABASE_URL"] = st.secrets["xata_db_url"]
os.environ["LLM_MODEL"] = st.secrets["llm_model"]
os.environ["LANGCHAIN_VERBOSE"] = str(st.secrets["langchain_verbose"])
os.environ["PASSWORD"] = st.secrets["password"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
os.environ["PINECONE_ENVIRONMENT"] = st.secrets["pinecone_environment"]
os.environ["PINECONE_INDEX"] = st.secrets["pinecone_index"]

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader, WikipediaLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import XataChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import DuckDuckGoSearchResults
from langchain.vectorstores import FAISS, Pinecone
from tenacity import retry, stop_after_attempt, wait_fixed
from xata.client import XataClient

import ui_config

ui = ui_config.create_ui_from_config()


llm_model = os.environ["LLM_MODEL"]
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

    :returns: An object representing the function calling chain, which is configured to generate structured output based on the provided JSON schema and chat prompt template.
    :rtype: object

    Function Behavior:
        - Defines a JSON schema for structured output that includes query information and date filters.
        - Creates a chat prompt template to instruct the underlying language model on how to generate the desired structured output.
        - Utilizes a language model for structured output generation.
        - Creates the function calling chain with 'create_structured_output_chain', passing the JSON schema, language model, and chat prompt template as arguments.

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
            "query": {
                "title": "Query",
                "description": "The next query extracted for a vector database semantic search from a chat history",
                "type": "string",
            },
            "arxiv_query": {
                "title": "arXiv Query",
                "description": "The next query for arXiv search extracted from a chat history in the format of a JSON object. Translate the query into accurate English if it is not already in English.",
                "type": "string",
            },
            "source": {
                "title": "Source Filter",
                "description": "Journal Name or Source extracted for a vector database semantic search, MUST be in upper case",
                "type": "string",
                "enum": [
                    "AGRICULTURE, ECOSYSTEMS & ENVIRONMENT",
                    "ANNUAL REVIEW OF ECOLOGY, EVOLUTION, AND SYSTEMATICS",
                    "ANNUAL REVIEW OF ENVIRONMENT AND RESOURCES",
                    "APPLIED CATALYSIS B: ENVIRONMENTAL",
                    "BIOGEOSCIENCES",
                    "BIOLOGICAL CONSERVATION",
                    "BIOTECHNOLOGY ADVANCES",
                    "CONSERVATION BIOLOGY",
                    "CONSERVATION LETTERS",
                    "CRITICAL REVIEWS IN ENVIRONMENTAL SCIENCE AND TECHNOLOGY",
                    "DIVERSITY AND DISTRIBUTIONS",
                    "ECOGRAPHY",
                    "ECOLOGICAL APPLICATIONS",
                    "ECOLOGICAL ECONOMICS",
                    "ECOLOGICAL MONOGRAPHS",
                    "ECOLOGY",
                    "ECOLOGY LETTERS",
                    "ECONOMIC SYSTEMS RESEARCH",
                    "ECOSYSTEM HEALTH AND SUSTAINABILITY",
                    "ECOSYSTEM SERVICES",
                    "ECOSYSTEMS",
                    "ENERGY & ENVIRONMENTAL SCIENCE",
                    "ENVIRONMENT INTERNATIONAL",
                    "ENVIRONMENTAL CHEMISTRY LETTERS",
                    "ENVIRONMENTAL HEALTH PERSPECTIVES",
                    "ENVIRONMENTAL POLLUTION",
                    "ENVIRONMENTAL SCIENCE & TECHNOLOGY",
                    "ENVIRONMENTAL SCIENCE & TECHNOLOGY LETTERS",
                    "ENVIRONMENTAL SCIENCE AND ECOTECHNOLOGY",
                    "ENVIRONMENTAL SCIENCE AND POLLUTION RESEARCH",
                    "EVOLUTION",
                    "FOREST ECOSYSTEMS",
                    "FRONTIERS IN ECOLOGY AND THE ENVIRONMENT",
                    "FRONTIERS OF ENVIRONMENTAL SCIENCE & ENGINEERING",
                    "FUNCTIONAL ECOLOGY",
                    "GLOBAL CHANGE BIOLOGY",
                    "GLOBAL ECOLOGY AND BIOGEOGRAPHY",
                    "GLOBAL ENVIRONMENTAL CHANGE",
                    "INTERNATIONAL SOIL AND WATER CONSERVATION RESEARCH",
                    "JOURNAL OF ANIMAL ECOLOGY",
                    "JOURNAL OF APPLIED ECOLOGY",
                    "JOURNAL OF BIOGEOGRAPHY",
                    "JOURNAL OF CLEANER PRODUCTION",
                    "JOURNAL OF ECOLOGY",
                    "JOURNAL OF ENVIRONMENTAL INFORMATICS",
                    "JOURNAL OF ENVIRONMENTAL MANAGEMENT",
                    "JOURNAL OF HAZARDOUS MATERIALS",
                    "JOURNAL OF INDUSTRIAL ECOLOGY",
                    "JOURNAL OF PLANT ECOLOGY",
                    "LANDSCAPE AND URBAN PLANNING",
                    "LANDSCAPE ECOLOGY",
                    "METHODS IN ECOLOGY AND EVOLUTION",
                    "MICROBIOME",
                    "MOLECULAR ECOLOGY",
                    "NATURE",
                    "NATURE CLIMATE CHANGE",
                    "NATURE COMMUNICATIONS",
                    "NATURE ECOLOGY & EVOLUTION",
                    "NATURE ENERGY",
                    "NATURE REVIEWS EARTH & ENVIRONMENT",
                    "NATURE SUSTAINABILITY",
                    "ONE EARTH",
                    "PEOPLE AND NATURE",
                    "PROCEEDINGS OF THE NATIONAL ACADEMY OF SCIENCES",
                    "PROCEEDINGS OF THE ROYAL SOCIETY B: BIOLOGICAL SCIENCES",
                    "RENEWABLE AND SUSTAINABLE ENERGY REVIEWS",
                    "RESOURCES, CONSERVATION AND RECYCLING",
                    "REVIEWS IN ENVIRONMENTAL SCIENCE AND BIO/TECHNOLOGY",
                    "SCIENCE",
                    "SCIENCE ADVANCES",
                    "SCIENCE OF THE TOTAL ENVIRONMENT",
                    "SCIENTIFIC DATA",
                    "SUSTAINABLE CITIES AND SOCIETY",
                    "SUSTAINABLE MATERIALS AND TECHNOLOGIES",
                    "SUSTAINABLE PRODUCTION AND CONSUMPTION",
                    "THE AMERICAN NATURALIST",
                    "THE INTERNATIONAL JOURNAL OF LIFE CYCLE ASSESSMENT",
                    "THE ISME JOURNAL",
                    "THE LANCET PLANETARY HEALTH",
                    "TRENDS IN ECOLOGY & EVOLUTION",
                    "WASTE MANAGEMENT",
                    "WATER RESEARCH",
                ],
            },
            "created_at": {
                "title": "Date Filter",
                "description": 'Date extracted for a vector database semantic search, in MongoDB\'s query and projection operators, in format like {"$gte": 1609459200.0, "$lte": 1640908800.0}',
                "type": "string",
            },
        },
        "required": ["query", "arxiv_query"],
    }

    prompt_func_calling_msgs = [
        SystemMessage(
            content="You are a world class algorithm for extracting the next query and filters for searching from a chat history. Make sure to answer in the correct structured format."
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


def search_pinecone(query: str, filters: dict = {}, top_k: int = 16):
    """
    Performs a similarity search on Pinecone's vector database based on a given query and optional date filter, and returns a list of relevant documents.

    :param query: The query to be used for similarity search in Pinecone's vector database.
    :type query: str
    :param created_at: The date filter to be applied in the search, specified in a format compatible with Pinecone's filtering options.
    :type created_at: str or None
    :param top_k: The number of top matching documents to return. Defaults to 16.
    :type top_k: int or None
    :returns: A list of dictionaries, each containing the content and source of the matched documents. The function returns an empty list if 'top_k' is set to 0.
    :rtype: list of dicts

    Function Behavior:
        - Initializes Pinecone with the specified API key and environment.
        - Conducts a similarity search based on the provided query and optional date filter.
        - Extracts and formats the relevant document information before returning.

    Exceptions:
        - This function relies on Pinecone and Python's os library. Exceptions could propagate if there are issues related to API keys, environment variables, or Pinecone initialization.
        - TypeError could be raised if the types of 'query', 'created_at', or 'top_k' do not match the expected types.

    Note:
        - Ensure the Pinecone API key and environment variables are set before running this function.
        - The function uses 'OpenAIEmbeddings' to initialize Pinecone's vector store, which should be compatible with the embeddings in the Pinecone index.
    """

    if top_k == 0:
        return []

    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    vectorstore = Pinecone.from_existing_index(
        index_name=os.environ["PINECONE_INDEX"],
        embedding=embeddings,
    )
    if filters:
        docs = vectorstore.similarity_search(query, k=top_k, filter=filters)
    else:
        docs = vectorstore.similarity_search(query, k=top_k)

    docs_list = []
    for doc in docs:
        date = datetime.fromtimestamp(doc.metadata["created_at"])
        formatted_date = date.strftime("%Y-%m")  # Format date as 'YYYY-MM'
        source_entry = "[{}. {}. {}. {}.]({})".format(
            doc.metadata["source_id"],
            doc.metadata["source"],
            doc.metadata["author"],
            formatted_date,
            doc.metadata["url"],
        )
        docs_list.append({"content": doc.page_content, "source": source_entry})

    return docs_list


def search_internet(query, top_k=4):
    """
    Performs an internet search based on the provided query using the DuckDuckGo search engine and returns a list of top results.

    :param query: The query string for the internet search.
    :type query: str
    :param top_k: The maximum number of top results to return. Defaults to 4.
    :type top_k: int or None.
    :returns: A list of dictionaries, each containing the snippet, title, and link of a search result. The function returns an empty list if 'top_k' is set to 0.
    :rtype: list of dicts

    Function Behavior:
        - Uses the DuckDuckGoSearchResults class to perform the search.
        - Parses the raw search results to extract relevant snippet, title, and link information.
        - Structures this information into a list of dictionaries and returns it.

    Exceptions:
        - This function relies on the DuckDuckGoSearchResults class, so exceptions might propagate from issues in that dependency.
        - TypeError could be raised if the types of 'query' or 'top_k' do not match the expected types.
    """

    if top_k == 0:
        return []

    search = DuckDuckGoSearchResults(num_results=top_k)

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


def wiki_query_func_calling_chain():
    """
    Identifies the language of a query for Wikipedia search and returns it as part of a structured output chain. This function relies on a specific JSON schema to define the structure of the output data.

    :returns: The structured output chain that can identify the language of a query for Wikipedia search, based on a pre-defined JSON schema.
    :rtype: object

    Function Behavior:
        - Initializes a ChatPromptTemplate with system messages and templates to guide the chat behavior.
        - Sets up a ChatOpenAI model for identifying the language.
        - Creates a structured output chain using the JSON schema, the language identification model, and the chat prompt template.
        - Returns the structured output chain, which can be executed later to obtain the language identifier for a Wikipedia search query.

    Exceptions:
        - This function relies on the create_structured_output_chain function, so any exceptions that occur in that function could propagate.
        - TypeError could be raised if internal components like llm_func_calling or prompt_func_calling do not match the expected types.

    Note:
        - The function is specifically tailored for language identification for Wikipedia queries and is not a generic language identification model.
        - It strictly follows a pre-defined language mapping, and the output language is one of the many predefined language codes.
    """

    func_calling_json_schema = {
        "title": "identify_query_language_to_search_Wikipedia",
        "description": "Accurately identifying the language of the query to search Wikipedia.",
        "type": "object",
        "properties": {
            "language": {
                "title": "Language",
                "description": "The accurate language of the query",
                "type": "string",
                "enum": [
                    "en",
                    "es",
                    "fr",
                    "de",
                    "ru",
                    "zh",
                    "pt",
                    "ar",
                    "it",
                    "ja",
                    "tr",
                    "id",
                    "simple",
                    "nl",
                    "pl",
                    "fa",
                    "he",
                    "vi",
                    "sv",
                    "ko",
                    "hi",
                    "uk",
                    "cs",
                    "ro",
                    "no",
                    "fi",
                    "hu",
                    "da",
                    "ca",
                    "th",
                    "bn",
                    "el",
                    "sr",
                    "bg",
                    "ms",
                    "hr",
                    "az",
                    "zh-yue",
                    "sk",
                    "sl",
                    "ta",
                    "arz",
                    "eo",
                    "sh",
                    "et",
                    "lt",
                    "ml",
                    "la",
                    "ur",
                    "af",
                    "mr",
                    "bs",
                    "sq",
                    "ka",
                    "eu",
                    "gl",
                    "hy",
                    "tl",
                    "be",
                    "kk",
                    "nn",
                    "ang",
                    "te",
                    "lv",
                    "ast",
                    "my",
                    "mk",
                    "ceb",
                    "sco",
                    "uz",
                    "als",
                    "zh-classical",
                    "is",
                    "mn",
                    "wuu",
                    "cy",
                    "kn",
                    "be-tarask",
                    "br",
                    "gu",
                    "an",
                    "bar",
                    "si",
                    "ne",
                    "sw",
                    "lb",
                    "zh-min-nan",
                    "jv",
                    "ckb",
                    "ga",
                    "war",
                    "ku",
                    "oc",
                    "nds",
                    "yi",
                    "ia",
                    "tt",
                    "fy",
                    "pa",
                    "azb",
                    "am",
                    "scn",
                    "lmo",
                    "gan",
                    "km",
                    "tg",
                    "ba",
                    "as",
                    "sa",
                    "ky",
                    "io",
                    "so",
                    "pnb",
                    "ce",
                    "vec",
                    "vo",
                    "mzn",
                    "or",
                    "cv",
                    "bh",
                    "pdc",
                    "hif",
                    "hak",
                    "mg",
                    "ht",
                    "ps",
                    "su",
                    "nap",
                    "qu",
                    "fo",
                    "bo",
                    "li",
                    "rue",
                    "se",
                    "nds-nl",
                    "gd",
                    "tk",
                    "yo",
                    "diq",
                    "pms",
                    "new",
                    "ace",
                    "vls",
                    "bat-smg",
                    "eml",
                    "cu",
                    "bpy",
                    "dv",
                    "hsb",
                    "sah",
                    "os",
                    "chr",
                    "sc",
                    "wa",
                    "szl",
                    "ha",
                    "ksh",
                    "bcl",
                    "nah",
                    "mt",
                    "co",
                    "ug",
                    "lad",
                    "cdo",
                    "pam",
                    "arc",
                    "crh",
                    "rm",
                    "zu",
                    "gv",
                    "frr",
                    "ab",
                    "got",
                    "iu",
                    "ie",
                    "xmf",
                    "cr",
                    "dsb",
                    "mi",
                    "gn",
                    "min",
                    "lo",
                    "sd",
                    "rmy",
                    "pcd",
                    "ilo",
                    "ext",
                    "sn",
                    "ig",
                    "nv",
                    "haw",
                    "csb",
                    "ay",
                    "jbo",
                    "frp",
                    "map-bms",
                    "lij",
                    "ch",
                    "vep",
                    "glk",
                    "tw",
                    "kw",
                    "bxr",
                    "wo",
                    "udm",
                    "av",
                    "pap",
                    "ee",
                    "cbk-zam",
                    "kv",
                    "fur",
                    "mhr",
                    "fiu-vro",
                    "bjn",
                    "roa-rup",
                    "gag",
                    "tpi",
                    "mai",
                    "stq",
                    "kab",
                    "bug",
                    "kl",
                    "nrm",
                    "mwl",
                    "bi",
                    "zea",
                    "ln",
                    "xh",
                    "myv",
                    "rw",
                    "nov",
                    "pfl",
                    "kaa",
                    "chy",
                    "roa-tara",
                    "pih",
                    "lfn",
                    "kg",
                    "bm",
                    "mrj",
                    "lez",
                    "za",
                    "om",
                    "ks",
                    "ny",
                    "krc",
                    "sm",
                    "st",
                    "pnt",
                    "dz",
                    "to",
                    "ary",
                    "tn",
                    "xal",
                    "gom",
                    "kbd",
                    "ts",
                    "rn",
                    "tet",
                    "mdf",
                    "ti",
                    "hyw",
                    "fj",
                    "tyv",
                    "ff",
                    "ki",
                    "ik",
                    "koi",
                    "lbe",
                    "jam",
                    "ss",
                    "lg",
                    "pag",
                    "tum",
                    "ve",
                    "ban",
                    "srn",
                    "ty",
                    "ltg",
                    "pi",
                    "sat",
                    "ady",
                    "olo",
                    "nso",
                    "sg",
                    "dty",
                    "din",
                    "tcy",
                    "gor",
                    "kbp",
                    "avk",
                    "lld",
                    "atj",
                    "inh",
                    "shn",
                    "nqo",
                    "mni",
                    "smn",
                    "mnw",
                    "dag",
                    "szy",
                    "gcr",
                    "awa",
                    "alt",
                    "shi",
                    "mad",
                    "skr",
                    "ami",
                    "trv",
                    "nia",
                    "tay",
                    "pwn",
                    "guw",
                    "pcm",
                    "kcg",
                    "blk",
                    "guc",
                    "anp",
                    "gur",
                    "fat",
                    "gpe",
                ],
            }
        },
        "required": ["language"],
    }

    prompt_func_calling_msgs = [
        SystemMessage(
            content="""You are a world class algorithm for accurately identifying the language of the query to search Wikipedia, strictly follow the language mapping: {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Russian": "ru", "Chinese": "zh", "Portuguese": "pt", "Arabic": "ar", "Italian": "it", "Japanese": "ja", "Turkish": "tr", "Indonesian": "id", "Simple English": "simple", "Dutch": "nl", "Polish": "pl", "Persian": "fa", "Hebrew": "he", "Vietnamese": "vi", "Swedish": "sv", "Korean": "ko", "Hindi": "hi", "Ukrainian": "uk", "Czech": "cs", "Romanian": "ro", "Norwegian": "no", "Finnish": "fi", "Hungarian": "hu", "Danish": "da", "Catalan": "ca", "Thai": "th", "Bangla": "bn", "Greek": "el", "Serbian": "sr", "Bulgarian": "bg", "Malay": "ms", "Croatian": "hr", "Azerbaijani": "az", "Cantonese": "zh-yue", "Slovak": "sk", "Slovenian": "sl", "Tamil": "ta", "Egyptian Arabic": "arz", "Esperanto": "eo", "Serbo-Croatian": "sh", "Estonian": "et", "Lithuanian": "lt", "Malayalam": "ml", "Latin": "la", "Urdu": "ur", "Afrikaans": "af", "Marathi": "mr", "Bosnian": "bs", "Albanian": "sq", "Georgian": "ka", "Basque": "eu", "Galician": "gl", "Armenian": "hy", "Tagalog": "tl", "Belarusian": "be", "Kazakh": "kk", "Norwegian Nynorsk": "nn", "Old English": "ang", "Telugu": "te", "Latvian": "lv", "Asturian": "ast", "Burmese": "my", "Macedonian": "mk", "Cebuano": "ceb", "Scots": "sco", "Uzbek": "uz", "Swiss German": "als", "Literary Chinese": "zh-classical", "Icelandic": "is", "Mongolian": "mn", "Wu Chinese": "wuu", "Welsh": "cy", "Kannada": "kn", "Belarusian (Taraškievica orthography)": "be-tarask", "Breton": "br", "Gujarati": "gu", "Aragonese": "an", "Bavarian": "bar", "Sinhala": "si", "Nepali": "ne", "Swahili": "sw", "Luxembourgish": "lb", "Min Nan Chinese": "zh-min-nan", "Javanese": "jv", "Central Kurdish": "ckb", "Irish": "ga", "Waray": "war", "Kurdish": "ku", "Occitan": "oc", "Low German": "nds", "Yiddish": "yi", "Interlingua": "ia", "Tatar": "tt", "Western Frisian": "fy", "Punjabi": "pa", "South Azerbaijani": "azb", "Amharic": "am", "Sicilian": "scn", "Lombard": "lmo", "Gan Chinese": "gan", "Khmer": "km", "Tajik": "tg", "Bashkir": "ba", "Assamese": "as", "Sanskrit": "sa", "Kyrgyz": "ky", "Ido": "io", "Somali": "so", "Western Punjabi": "pnb", "Chechen": "ce", "Venetian": "vec", "Volapük": "vo", "Mazanderani": "mzn", "Odia": "or", "Chuvash": "cv", "Bhojpuri": "bh", "Pennsylvania German": "pdc", "Fiji Hindi": "hif", "Hakka Chinese": "hak", "Malagasy": "mg", "Haitian Creole": "ht", "Pashto": "ps", "Sundanese": "su", "Neapolitan": "nap", "Quechua": "qu", "Faroese": "fo", "Tibetan": "bo", "Limburgish": "li", "Rusyn": "rue", "Northern Sami": "se", "Low Saxon": "nds-nl", "Scottish Gaelic": "gd", "Turkmen": "tk", "Yoruba": "yo", "Zazaki": "diq", "Piedmontese": "pms", "Newari": "new", "Achinese": "ace", "West Flemish": "vls", "Samogitian": "bat-smg", "Emiliano-Romagnolo": "eml", "Church Slavic": "cu", "Bishnupriya": "bpy", "Divehi": "dv", "Upper Sorbian": "hsb", "Yakut": "sah", "Ossetic": "os", "Cherokee": "chr", "Sardinian": "sc", "Walloon": "wa", "Silesian": "szl", "Hausa": "ha", "Colognian": "ksh", "Central Bikol": "bcl", "Nāhuatl": "nah", "Maltese": "mt", "Corsican": "co", "Uyghur": "ug", "Ladino": "lad", "Min Dong Chinese": "cdo", "Pampanga": "pam", "Aramaic": "arc", "Crimean Tatar": "crh", "Romansh": "rm", "Zulu": "zu", "Manx": "gv", "Northern Frisian": "frr", "Abkhazian": "ab", "Gothic": "got", "Inuktitut": "iu", "Interlingue": "ie", "Mingrelian": "xmf", "Cree": "cr", "Lower Sorbian": "dsb", "Māori": "mi", "Guarani": "gn", "Minangkabau": "min", "Lao": "lo", "Sindhi": "sd", "Vlax Romani": "rmy", "Picard": "pcd", "Iloko": "ilo", "Extremaduran": "ext", "Shona": "sn", "Igbo": "ig", "Navajo": "nv", "Hawaiian": "haw", "Kashubian": "csb", "Aymara": "ay", "Lojban": "jbo", "Arpitan": "frp", "Basa Banyumasan": "map-bms", "Ligurian": "lij", "Chamorro": "ch", "Veps": "vep", "Gilaki": "glk", "Twi": "tw", "Cornish": "kw", "Russia Buriat": "bxr", "Wolof": "wo", "Udmurt": "udm", "Avaric": "av", "Papiamento": "pap", "Ewe": "ee", "Chavacano": "cbk-zam", "Komi": "kv", "Friulian": "fur", "Eastern Mari": "mhr", "Võro": "fiu-vro", "Banjar": "bjn", "Aromanian": "roa-rup", "Gagauz": "gag", "Tok Pisin": "tpi", "Maithili": "mai", "Saterland Frisian": "stq", "Kabyle": "kab", "Buginese": "bug", "Kalaallisut": "kl", "Norman": "nrm", "Mirandese": "mwl", "Bislama": "bi", "Zeelandic": "zea", "Lingala": "ln", "Xhosa": "xh", "Erzya": "myv", "Kinyarwanda": "rw", "Novial": "nov", "Palatine German": "pfl", "Kara-Kalpak": "kaa", "Cheyenne": "chy", "Tarantino": "roa-tara", "Norfuk / Pitkern": "pih", "Lingua Franca Nova": "lfn", "Kongo": "kg", "Bambara": "bm", "Western Mari": "mrj", "Lezghian": "lez", "Zhuang": "za", "Oromo": "om", "Kashmiri": "ks", "Nyanja": "ny", "Karachay-Balkar": "krc", "Samoan": "sm", "Southern Sotho": "st", "Pontic": "pnt", "Dzongkha": "dz", "Tongan": "to", "Moroccan Arabic": "ary", "Tswana": "tn", "Kalmyk": "xal", "Goan Konkani": "gom", "Kabardian": "kbd", "Tsonga": "ts", "Rundi": "rn", "Tetum": "tet", "Moksha": "mdf", "Tigrinya": "ti", "Western Armenian": "hyw", "Fijian": "fj", "Tuvinian": "tyv", "Fula": "ff", "Kikuyu": "ki", "Inupiaq": "ik", "Komi-Permyak": "koi", "Lak": "lbe", "Jamaican Creole English": "jam", "Swati": "ss", "Ganda": "lg", "Pangasinan": "pag", "Tumbuka": "tum", "Venda": "ve", "Balinese": "ban", "Sranan Tongo": "srn", "Tahitian": "ty", "Latgalian": "ltg", "Pali": "pi", "Santali": "sat", "Adyghe": "ady", "Livvi-Karelian": "olo", "Northern Sotho": "nso", "Sango": "sg", "Doteli": "dty", "Dinka": "din", "Tulu": "tcy", "Gorontalo": "gor", "Kabiye": "kbp", "Kotava": "avk", "Ladin": "lld", "Atikamekw": "atj", "Ingush": "inh", "Shan": "shn", "N’Ko": "nqo", "Manipuri": "mni", "Inari Sami": "smn", "Mon": "mnw", "Dagbani": "dag", "Sakizaya": "szy", "Guianan Creole": "gcr", "Awadhi": "awa", "Southern Altai": "alt", "Tachelhit": "shi", "Madurese": "mad", "Saraiki": "skr", "Amis": "ami", "Taroko": "trv", "Nias": "nia", "Tayal": "tay", "Paiwan": "pwn", "Gun": "guw", "Nigerian Pidgin": "pcm", "Tyap": "kcg", "Pa"O": "blk", "Wayuu": "guc", "Angika": "anp", "Frafra": "gur", "Fanti": "fat", "Ghanaian Pidgin": "gpe"}"""
        ),
        HumanMessage(content="The query:"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]

    prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

    llm_func_calling = ChatOpenAI(model_name=llm_model, temperature=0, streaming=False)

    query_func_calling_chain = create_structured_output_chain(
        output_schema=func_calling_json_schema,
        llm=llm_func_calling,
        prompt=prompt_func_calling,
        verbose=langchain_verbose,
    )

    return query_func_calling_chain


def search_wiki(query: str, top_k=16) -> list:
    """
    Searches Wikipedia for relevant documents based on the given query and returns the top k results.

    :param query: The search query string.
    :type query: str
    :param top_k: The maximum number of results to return. Defaults to 16.
    :type top_k: int
    :return: A list of dictionaries, each containing the page content and source of a search result. Returns an empty list if 'top_k' is set to 0.
    :rtype: list

    Function Behavior:
        - Identifies the language of the query using the :func: `wiki_query_func_calling_chain` function.
        - Uses the `WikipediaLoader` class to load Wikipedia documents relevant to the query.
        - Splits the text into manageable chunks using `RecursiveCharacterTextSplitter`.
        - Embeds the text chunks using `OpenAIEmbeddings` and indexes them with FAISS.
        - Performs a similarity search on the query to retrieve the top k relevant documents.

    Exceptions:
        - KeyError: If the 'language' key is not present in the result from `wiki_query_func_calling_chain`.
        - Any exceptions that may be raised by dependent classes or functions like `WikipediaLoader`, `RecursiveCharacterTextSplitter`, etc.
    """

    if top_k == 0:
        return []

    language = wiki_query_func_calling_chain().run(query)["language"]
    docs = WikipediaLoader(
        query=query, lang=language, load_max_docs=2, load_all_available_meta=True
    ).load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=20
    )
    chunks = []

    for doc in docs:
        chunk = text_splitter.create_documents(
            [doc.page_content],
            metadatas=[
                {
                    "source": "[{}]({})".format(
                        doc.metadata["title"], doc.metadata["source"]
                    )
                }
            ],
        )
        chunks.extend(chunk)

    embeddings = OpenAIEmbeddings()
    faiss_db = FAISS.from_documents(chunks, embeddings)

    result_docs = faiss_db.similarity_search(query, k=top_k)

    docs_list = []

    for doc in result_docs:
        source_entry = doc.metadata["source"]
        docs_list.append({"content": doc.page_content, "source": source_entry})

    return docs_list


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_file_to_stream(url):
    """
    Downloads a file from a given URL and returns its content as a binary stream. The function will retry up to 3 times with a 2-second wait in case of failure.

    :param url: The URL of the file to download.
    :type url: str
    :returns: A BytesIO object containing the file content.
    :rtype: io.BytesIO

    Function Behavior:
        - Sets up headers with a fake User-Agent to send the HTTP GET request.
        - Sends an HTTP GET request to the specified URL, streaming the response.
        - Checks the HTTP status code and raises an exception if it's not 200.
        - Reads the content in chunks and writes it to a BytesIO object.
        - Seeks back to the beginning of the BytesIO object before returning it.

    Exceptions:
        - requests.exceptions.RequestException could be raised if the HTTP GET request fails.
        - TypeError could be raised if the type of 'url' does not match the expected type.

    Note:
        - This function uses the `retry` decorator to automatically handle retries.
    """

    # 使用伪造的User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
    }

    response = requests.get(url, stream=True, headers=headers)

    # 检查状态码
    if response.status_code != 200:
        response.raise_for_status()  # 引发异常，如果有错误

    file_stream = io.BytesIO()

    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            file_stream.write(chunk)

    # 将文件指针重置到流的开头
    file_stream.seek(0)
    return file_stream


def parse_paper(pdf_stream):
    """
    Parses the text content of a PDF paper using pdfplumber. This function aims to extract the main text body from a paper, skipping headers, footers, and other irrelevant sections like "Acknowledgements" and "References."

    :param pdf_stream: The byte stream of the PDF to be parsed.
    :type pdf_stream: io.BytesIO or file-like object
    :returns: The extracted main text of the PDF as a single string.
    :rtype: str

    Function Behavior:
        - Reads the PDF from the byte stream using pdfplumber.
        - Goes through each page to identify and collect sentences that are part of the main text.
        - Takes into account special cases for the first page and reference section.
        - Ignores text if it falls under headers, footers, and miscellaneous sections like "References" and "Acknowledgements."
        - Utilizes helper function 'visitor_body' to check whether each piece of text should be included based on its coordinates and other attributes.

    Local Functions:
        - visitor_body(text, isfirstpage, x, top, bottom, fontSize, ismisc): A helper function that decides whether the given text should be included in the main text body based on its attributes and location on the page.

    Exceptions:
        - This function relies on pdfplumber for parsing PDF, so any exception thrown by pdfplumber can propagate. Such exceptions might include but are not limited to:
            - PDFSyntaxError: If the PDF has syntactical errors.
            - FileNotFoundError: If the pdf_stream is a file path and the file doesn't exist.

    Note:
        - The function assumes that the main text has a consistent font size throughout the paper, and uses this assumption for text extraction.
        - The function treats the first page separately as it may contain unique elements like title and abstract.
        - It also switches to a mode where it ignores most text once it detects sections like "References" or "Acknowledgements."
    """

    # logging.info("Parsing paper")
    pdf_obj = pdfplumber.open(pdf_stream)
    number_of_pages = len(pdf_obj.pages)
    # logging.info(f"Total number of pages: {number_of_pages}")
    full_text = ""
    ismisc = False
    for i in range(number_of_pages):
        page = pdf_obj.pages[i]
        if i == 0:
            isfirstpage = True
        else:
            isfirstpage = False

        page_text = []
        sentences = []
        processed_text = []

        def visitor_body(text, isfirstpage, x, top, bottom, fontSize, ismisc):
            # ignore header/footer
            if isfirstpage:
                if (top > 200 and bottom < 720) and (len(text.strip()) > 1):
                    sentences.append(
                        {
                            "fontsize": fontSize,
                            "text": " " + text.strip().replace("\x03", ""),
                            "x": x,
                            "y": top,
                        }
                    )
            else:  # not first page
                if (
                    (top > 70 and bottom < 720)
                    and (len(text.strip()) > 1)
                    and not ismisc
                ):  # main text region
                    sentences.append(
                        {
                            "fontsize": fontSize,
                            "text": " " + text.strip().replace("\x03", ""),
                            "x": x,
                            "y": top,
                        }
                    )
                elif (top > 70 and bottom < 720) and (len(text.strip()) > 1) and ismisc:
                    pass

        extracted_words = page.extract_words(
            x_tolerance=1,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            horizontal_ltr=True,
            vertical_ttb=True,
            extra_attrs=["fontname", "size"],
            split_at_punctuation=False,
        )

        # Treat the first page, main text, and references differently, specifically targeted at headers
        # Define a list of keywords to ignore
        # Online is for Nauture papers
        keywords_for_misc = [
            "References",
            "REFERENCES",
            "Bibliography",
            "BIBLIOGRAPHY",
            "Acknowledgements",
            "ACKNOWLEDGEMENTS",
            "Acknowledgments",
            "ACKNOWLEDGMENTS",
            "Acknowledgement",
            "参考文献",
            "致谢",
            "謝辞",
            "謝",
            "Online",
        ]

        prev_word_size = None
        prev_word_font = None
        # Loop through the extracted words
        for extracted_word in extracted_words:
            # Strip the text and remove any special characters
            text = extracted_word["text"].strip().replace("\x03", "")

            # Check if the text contains any of the keywords to ignore
            if any(keyword in text for keyword in keywords_for_misc) and (
                prev_word_size != extracted_word["size"]
                or prev_word_font != extracted_word["fontname"]
            ):
                ismisc = True

            prev_word_size = extracted_word["size"]
            prev_word_font = extracted_word["fontname"]

            # Call the visitor_body function with the relevant arguments
            visitor_body(
                text,
                isfirstpage,
                extracted_word["x0"],
                extracted_word["top"],
                extracted_word["bottom"],
                extracted_word["size"],
                ismisc,
            )

        if sentences:
            for sentence in sentences:
                page_text.append(sentence)

        blob_font_sizes = []
        blob_font_size = None
        blob_text = ""
        processed_text = ""
        tolerance = 1

        # Preprocessing for main text font size
        if page_text != []:
            if len(page_text) == 1:
                blob_font_sizes.append(page_text[0]["fontsize"])
            else:
                for t in page_text:
                    blob_font_sizes.append(t["fontsize"])
            blob_font_size = Counter(blob_font_sizes).most_common(1)[0][0]

        if page_text != []:
            if len(page_text) == 1:
                if (
                    blob_font_size - tolerance
                    <= page_text[0]["fontsize"]
                    <= blob_font_size + tolerance
                ):
                    processed_text += page_text[0]["text"]
                    # processed_text.append({"text": page_text[0]["text"], "page": i + 1})
            else:
                for t in range(len(page_text)):
                    if (
                        blob_font_size - tolerance
                        <= page_text[t]["fontsize"]
                        <= blob_font_size + tolerance
                    ):
                        blob_text += f"{page_text[t]['text']}"
                        if len(blob_text) >= 500:  # set the length of a data chunk
                            processed_text += blob_text
                            # processed_text.append({"text": blob_text, "page": i + 1})
                            blob_text = ""
                        elif t == len(page_text) - 1:  # last element
                            processed_text += blob_text
                            # processed_text.append({"text": blob_text, "page": i + 1})
            full_text += processed_text

    # logging.info("Done parsing paper")
    return full_text


def search_arxiv_docs(query: str, top_k=16) -> list:
    """
    Searches for academic papers on arxiv.org based on a query and returns the top-k most relevant results.

    :param query: The search query for arXiv.
    :type query: str
    :param top_k: The maximum number of top results to return. Defaults to 16.
    :type top_k: int or None
    :returns: A list of dictionaries, each containing the content and source information of a search result. The function returns an empty list if 'top_k' is set to 0.
    :rtype: list of dicts

    Function Behavior:
        - Checks if 'top_k' is zero and, if so, returns an empty list.
        - Uses the arXiv API to perform the initial search and get a list of relevant papers.
        - Downloads the PDFs and extracts the text content.
        - Splits the text content into smaller chunks for better handling.
        - Embeds the chunks and performs a similarity search.
        - Structures the similarity search results into a list of dictionaries and returns it.

    Exceptions:
        - arxiv.ArxivException: Raised if the search query to arXiv API fails.
        - requests.exceptions.RequestException: May be propagated from the 'download_file_to_stream' function.
        - TypeError: Could be raised if the types of 'query' or 'top_k' do not match the expected types.

    Note:
        - This function relies on several external libraries and methods like arXiv API, FAISS, and OpenAIEmbeddings.
    """

    if top_k == 0:
        return []

    docs = arxiv.Search(
        query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance
    ).results()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=20
    )
    chunks = []

    for doc in docs:
        pdf_stream = download_file_to_stream(doc.pdf_url)

        page_content = parse_paper(pdf_stream)
        authors = ", ".join(str(author) for author in doc.authors)
        date = doc.published.strftime("%Y-%m")

        source = "[{}. {}. {}.]({})".format(
            authors,
            doc.title,
            date,
            doc.entry_id,
        )

        chunk = text_splitter.create_documents(
            [page_content], metadatas=[{"source": source}]
        )

        chunks.extend(chunk)

    if chunks == []:
        return []

    embeddings = OpenAIEmbeddings()
    faiss_db = FAISS.from_documents(chunks, embeddings)

    result_docs = faiss_db.similarity_search(query, k=top_k)
    docs_list = []
    for doc in result_docs:
        source_entry = doc.metadata["source"]
        docs_list.append({"content": doc.page_content, "source": source_entry})

    return docs_list


def get_faiss_db(uploaded_files):
    """
    Creates a FAISS database from a list of uploaded files.

    :param uploaded_files: List of uploaded file objects.
    :type uploaded_files: list of file-like objects
    :returns: A FAISS database containing the embeddings of the chunks from the uploaded files' content.
    :rtype: FAISS database object

    Function Behavior:
        - Iterates through the list of uploaded files.
        - Reads the content of each uploaded file and processes it into smaller chunks.
        - Embeds the chunks using OpenAIEmbeddings.
        - Creates a FAISS database from the embeddings of the chunks.
        - If the chunks list is empty, a warning message is displayed and the function terminates.

    Exceptions:
        - Could raise exceptions related to file reading or invalid file format.
        - An exception may propagate from the FAISS.from_documents method.
        - TypeError could be raised if the types of 'uploaded_files' do not match the expected types.

    Note:
        - This function relies on the FAISS library for creating the database and OpenAIEmbeddings for generating embeddings.
    """

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=20
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


def search_uploaded_docs(query, top_k=16):
    """
    Searches the FAISS database for similar documents based on the provided query and returns a list of top results.

    :param query: The query string for the similarity search.
    :type query: str
    :param top_k: The maximum number of top results to return. Defaults to 16.
    :type top_k: int or None
    :returns: A list of dictionaries, each containing the content and source of a matching document. The function returns an empty list if 'top_k' is set to 0.
    :rtype: list of dicts

    Function Behavior:
        - Retrieves the FAISS database from the Streamlit session state.
        - Performs a similarity search in the FAISS database based on the query.
        - Structures the search results into a list of dictionaries and returns it.

    Exceptions:
        - TypeError could be raised if the types of 'query' or 'top_k' do not match the expected types.
    """

    if top_k == 0:
        return []

    docs = st.session_state["faiss_db"].similarity_search(query, k=top_k)
    docs_list = []
    for doc in docs:
        source_entry = doc.metadata["source"]
        docs_list.append({"content": doc.page_content, "source": source_entry})

    return docs_list


def chat_history_chain():
    """
    Creates and returns a Large Language Model (LLM) chain configured to produce highly concise and well-organized chat history.

    :return: A configured LLM chain object for producing concise chat histories.
    :rtype: Object

    Function Behavior:
        - Initializes a ChatOpenAI instance for a specific language model.
        - Configures a prompt template asking for a highly concise and well-organized chat history.
        - Constructs and returns an LLMChain instance, which uses the configured language model and prompt template.

    Exceptions:
        - Exceptions could propagate from underlying dependencies like the ChatOpenAI or LLMChain classes.
        - TypeError could be raised if internal configurations within the function do not match the expected types.
    """

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

    llm_chat = ChatOpenAI(
        model=llm_model,
        temperature=0,
        streaming=True,
        verbose=langchain_verbose,
    )

    template = """You MUST ONLY responese to science-related quests.
    DO NOT return any information on politics, ethnicity, gender, national sovereignty, or other sensitive topics.
    {input}"""

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
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "avatar": ui.chat_ai_avatar,
                "content": ui.chat_ai_welcome,
            }
        ]
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"], avatar=msg["avatar"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)

    return execute


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
        - Transforms it into a dictionary containing the role ('user' or 'assistant') and the content of the message.
    """
    if isinstance(history, HumanMessage):
        return {
            "role": "user",
            "avatar": ui.chat_user_avatar,
            "content": history.content,
        }
    elif isinstance(history, AIMessage):
        return {
            "role": "assistant",
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
