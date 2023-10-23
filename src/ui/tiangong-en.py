ui_data = {
    "need_fixed_passwd": False,
    "wix_login_title": "TianGong Login",
    "wix_login_username_label": "Email Address",
    "wix_login_password_label": "Password",
    "wix_login_button_label": "Log in",
    "wix_signup_button_label": "Or Sign Up on Our Website 🌐",
    "wix_login_error_text": "Login failed, please check your username and password.",
    "wix_signup_button_url": "https://www.example.com/",
    "wix_login_error_icon": "🚨",
    "theme": {
        "primaryColor": "#82318E",
    },
    "page_title": "TianGong Chat",
    "page_icon": "src/static/favicon.ico",
    "page_markdown": """
      <style>
          body {
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
          }
          a {
              text-decoration: none;
          }
          a:hover {
              text-decoration: none;
          }
          div[data-testid="stDecoration"] {
              background-image: linear-gradient(90deg, #82318E, #C2B6B2);
          }
          div[data-testid="stImage"] {
              min-width: 50px;
              max-width: 50px;
          }
          section[data-testid="stSidebar"] {
              min-width: 360px;
              max-width: 480px;
          }
          .css-164nlkn {
              text-align: right;
          }
          .thusoetext {
              width: 200px;
              height: 50px;
              background-image: url('app/static/thu_soe_text.png');
              background-size: 180px 50px;
              background-repeat: no-repeat;
              background-position: center;
              opacity: 0.5;
          }
          .thulogo {
              width: 50px;
              height: 50px;
              background-image: url('app/static/thu_logo.png');
              background-size: 50px 50px;
              background-repeat: no-repeat;
              background-position: center;
              opacity: 0.5;
          }
          .soelogo {
              width: 50px;
              height: 50px;
              background-image: url('app/static/soe_logo.png');
              background-size: 45px 45px;
              background-repeat: no-repeat;
              background-position: center;
              opacity: 0.5;
          }
          .text-sm {
              font-size: .875rem;
              line-height: 1.25rem
          }
      </style>
      """,
    "sidebar_image": "src/static/logo.png",
    "sidebar_title": "TianGong Chat",
    "sidebar_subheader": "AI for Science 🔬 - Intelligent Chatbot",
    "sidebar_welcome_text": """Hi, **{username}**! You're currently on the **{subscription}** plan.""",
    "sidebar_markdown": """
        <div style="position: fixed; left:25px; bottom: 10px; height:110px; width: 315px; font-size: 14px; font-weight: bold; background-color: #F0F2F6;">
            <table style="border: 0; position: absolute; bottom: 35px;">
                <tr style="border: 0; width:315px; ">
                    <td class="thulogo" style="border: 0;"></td>
                    <td class="soelogo" style="border: 0;"></td>
                    <td class="thusoetext" style="border: 0;"></td>
                </tr>
                <tr style="border: 0;">
                    <td colspan="3" style="border: 0; padding: 0;"><p class="text-sm" style="opacity: 0.5;text-align: left; margin: 0;"><a style="width: 315px; color: black; display: block" href="https://mingxu.tiangong.world" target="_blank">Environmental Data Science and Systems Engineering Research Team</a></p>
                    </td>
                </tr>
            </table>
            <table style="border: 0; position: absolute; bottom: 5px; width:315px;">
                <tr style="border: 0;">
                    <td style="border: 0; text-align: center; padding: 0;"><p class="text-sm" style="margin: 0; opacity: 0.5;"><a style="color: black" href="https://ai-en.tiangong.world/">TianGong AI 🧠</a></p></td>
                    <td style="border: 0; text-align: center; padding: 0;">
                        <p class="text-sm" style="opacity: 0.5; margin: 0;"><a style="color: black" href="mailto: gpt@tiangong.world">Technical Support 📧</a></p>
                    </td>
                </tr>
            </table>
        </div>
        """,
    "sidebar_expander_title": "Advanced Search Settings:",
    "search_knowledge_base_checkbox_label": "Knowledge Base",
    "search_internet_checkbox_label": "Internet",
    "search_wikipedia_checkbox_label": "Wikipedia",
    "search_arxiv_checkbox_label": "arXiv",
    "search_docs_checkbox_label": "Documents",
    "search_docs_options": "Options:",
    "search_docs_options_isolated": "Isolated",
    "search_docs_options_combined": "Combined",
    "sidebar_file_uploader_title": "Documents to analyze:",
    "sidebar_file_uploader_spinner": "Analyzing...",
    "sidebar_file_uploader_error": "None of the files can be analyzed, please check the format!",
    "sidebar_instructions": """*I am a Retrieval Augmentation Generation (RAG) tool designed for academic and professional documents. I use the information provided in your prompt to **search** for relevant documents and then generate responses based on them*.
### 🌟 Ideal Questions:
1. *What is the methods and mechanisms of carbon reduction?*
2. *How to conduct a dynamic material flow analysis? Please refer to EST journal since 2021. ([Available Journal List](https://ai-en.tiangong.world/journal-list/))*
### 🚫 Avoid Questions:
1. *Hello!? What can you do?*
2. *Help me write/translate an essay on sustainable development. (For this kind of job you may refer to [TianGong Pro Agent](https://ai.tiangong.world/pro-intro/))*
3. *What **articles** have been published in the RCR journal since 2015? (**DO NOT** search for articles **BUT** detailed information)*""",
    "current_chat_title": "Chat History :",
    "chat_ai_avatar": "src/static/logo.png",
    "chat_user_avatar": "src/static/user.png",
    "chat_ai_welcome": "How can I help you?",
    "chat_human_placeholder": "Ask me anything, in any language you like!",
    "sidebar_newchat_button_label": "New Chat",
    "sidebar_delete_button_label": "Delete Chat",
}
