"""utilities used in the app"""
import io
import os
import re
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

from langchain import LLMChain, PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader, WikipediaLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import XataChatMessageHistory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
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


def check_password():
    """Returns `True` if the user had the correct password."""

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
    func_calling_json_schema = {
        "title": "get_querys_and_filters_to_search",
        "description": "Extract the next queries and filters for searching from a chat history.",
        "type": "object",
        "properties": {
            "query": {
                "title": "Query",
                "description": "The next query extracted for a vector database semantic search from a chat history in the format of a JSON object",
                "type": "string",
            },
            "arxiv_query": {
                "title": "arXiv Query",
                "description": "The next query for arXiv search extracted from a chat history in the format of a JSON object. Translate the query into accurate English if it is not already in English.",
                "type": "string",
            },
            "created_at": {
                "title": "Date Filters",
                "description": 'Date extracted for a vector database semantic search from a chat history, in MongoDB\'s query and projection operators, in format like {"$gte": 1609459200.0, "$lte": 1640908800.0}',
                "type": "string",
            },
        },
        "required": ["query", "arxiv_query"],
    }

    prompt_func_calling_msgs = [
        SystemMessage(
            content="You are a world class algorithm for extracting the next query and filters for searching from a chat history. Make sure to answer in the correct structured format"
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


def search_pinecone(query, created_at, top_k=16):
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


def search_internet(query, top_k=4):
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
    """Search Wikipedia for results."""
    if top_k == 0:
        return []

    language = wiki_query_func_calling_chain().run(query)["language"]
    docs = WikipediaLoader(
        query=query, lang=language, load_max_docs=2, load_all_available_meta=True
    ).load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=220, chunk_overlap=20
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
    """Search arxiv.org for results."""
    if top_k == 0:
        return []

    docs = arxiv.Search(
        query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance
    ).results()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=220, chunk_overlap=20
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


def seach_uploaded_docs(query, top_k=16):
    if top_k == 0:
        return []

    docs = st.session_state["faiss_db"].similarity_search(query, k=top_k)
    docs_list = []
    for doc in docs:
        source_entry = doc.metadata["source"]
        docs_list.append({"content": doc.page_content, "source": source_entry})

    return docs_list


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
    chat_history = XataChatMessageHistory(
        session_id=_session_id,
        api_key=os.environ["XATA_API_KEY"],
        db_url=os.environ["XATA_DATABASE_URL"],
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
