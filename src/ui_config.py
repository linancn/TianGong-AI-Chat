import importlib
import os
from dataclasses import dataclass

import toml

ui_config_file = f"""ui.{os.environ.get("ui",default="tiangong-en")}"""
config_module = importlib.import_module(ui_config_file)
ui_data = config_module.ui_data


@dataclass
class Theme:
    base: str
    primaryColor: str
    backgroundColor: str
    secondaryBackgroundColor: str
    textColor: str
    font: str


@dataclass
class UI:
    need_fixed_passwd: bool
    wix_login_title: str
    wix_login_username_label: str
    wix_login_password_label: str
    wix_login_button_label: str
    wix_signup_button_label: str
    wix_signup_button_url: str
    wix_login_error_text: str
    wix_login_error_timeout: str
    wix_login_error_icon: str
    wix_login_no_active_subscription_text: str
    wix_login_no_active_subscription_icon: str
    theme: Theme
    page_title: str
    page_icon: str
    page_markdown: str
    sidebar_image: str
    sidebar_title: str
    sidebar_subheader: str
    sidebar_welcome_text: str
    sidebar_markdown: str
    sidebar_expander_title: str
    search_journal_paper_checkbox_label: str
    search_internet_checkbox_label: str
    search_report_checkbox_label: str
    search_standard_checkbox_label: str
    search_patent_checkbox_label: str
    search_docs_checkbox_label: str
    search_docs_options: str
    search_docs_options_isolated: str
    search_docs_options_combined: str
    sidebar_file_uploader_title: str
    sidebar_file_uploader_spinner: str
    sidebar_file_uploader_error: str
    sidebar_instructions: str
    sidebar_newchat_button_label: str
    sidebar_delete_button_label: str
    sidebar_newchat_label: str
    current_chat_title: str
    chat_ai_avatar: str
    chat_user_avatar: str
    chat_ai_welcome: str
    chat_human_placeholder: str
    chat_error_message: str
    wix_login_wait:str


def create_ui_from_config():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.abspath(
        os.path.join(current_dir, "../.streamlit/config.toml")
    )
    # 读取文件
    with open(config_path, "r") as file:
        data = toml.load(file)

    # 替换内容
    if data["theme"] != ui_data["theme"]:
        data["theme"] = ui_data["theme"]
        # 写入文件
        with open(config_path, "w") as file:
            toml.dump(data, file)

    return UI(**ui_data)
