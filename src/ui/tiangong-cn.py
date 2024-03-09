ui_data = {
    "need_fixed_passwd": False,
    "wix_login_title": "天工登录",
    "wix_login_username_label": "电子邮件地址",
    "wix_login_password_label": "密码",
    "wix_login_button_label": "登录",
    "wix_signup_button_label": "或在我们的网站上注册 🌐",
    "wix_login_error_text": "登录失败，请检查您的用户名和密码。",
    "wix_signup_button_url": "https://www.example.com/",
    "wix_login_error_icon": "🚨",
    "theme": {"primaryColor": "#82318E"},
    "page_title": "天工聊天",
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
    "sidebar_title": "天工Chat",
    "sidebar_subheader": "AI for Science 🔬 - 您的私人智能助理",
    "sidebar_welcome_text": """嗨，**{username}**！您当前处于**{subscription}**计划。""",
    "sidebar_markdown": """
        <div style="position: fixed; left:25px; bottom: 10px; height:110px; width: 349px; font-size: 14px; font-weight: bold; background-color: #F0F2F6;">
            <table style="border: 0; position: absolute; bottom: 35px;">
                <tr style="border: 0; width:315px; ">
                    <td class="thulogo" style="border: 0;"></td>
                    <td class="soelogo" style="border: 0;"></td>
                    <td class="thusoetext" style="border: 0;"></td>
                </tr>
                <tr style="border: 0;">
                    <td colspan="3" style="border: 0; padding: 0;"><p class="text-sm" style="opacity: 0.5;text-align: center; margin: 0;"><a style="width: 315px; color: black; display: block" href="https://mingxu.tiangong.world" target="_blank">环境数据科学与系统工程研究团队</a></p>
                    </td>
                </tr>
            </table>
            <table style="border: 0; position: absolute; bottom: 5px; width:315px;">
                <tr style="border: 0;">
                    <td style="border: 0; text-align: center; padding: 0;"><p class="text-sm" style="margin: 0; opacity: 0.5;"><a style="color: black" href="https://ai-en.tiangong.world/">天工AI 🧠</a></p></td>
                    <td style="border: 0; text-align: center; padding: 0;">
                        <p class="text-sm" style="opacity: 0.5; margin: 0;"><a style="color: black" href="mailto: gpt@tiangong.world">技术支持 📧</a></p>
                    </td>
                </tr>
            </table>
        </div>
        """,
    "sidebar_expander_title": "高级搜索设置：",
    "search_knowledge_base_checkbox_label": "知识库",
    "search_internet_checkbox_label": "互联网",
    "search_wikipedia_checkbox_label": "维基百科",
    "search_arxiv_checkbox_label": "arXiv",
    "search_docs_checkbox_label": "文档",
    "search_docs_options": "选项：",
    "search_docs_options_isolated": "独立",
    "search_docs_options_combined": "组合",
    "sidebar_file_uploader_title": "待分析的文档：",
    "sidebar_file_uploader_spinner": "正在分析...",
    "sidebar_file_uploader_error": "所有文件均无法分析，请检查格式！",
    "sidebar_instructions": """*我是一个为学术设计的检索增强生成（RAG）工具。我使用您在提示中提供的信息来**搜索**相关信息，然后基于它们生成结果*。""",
    "current_chat_title": "聊天历史：",
    "chat_ai_avatar": "src/static/logo.png",
    "chat_user_avatar": "src/static/user.png",
    "chat_ai_welcome": "您好！有什么可以帮您？",
    "chat_human_placeholder": "您可以用任何语言向我提问！",
    "sidebar_newchat_button_label": "新建",
    "sidebar_delete_button_label": "删除",
    "sidebar_newchat_label": "新对话",
    "chat_error_message": "哎呀，我们正面临极高的访问量。请稍后再试。",
    "wix_login_wait": "请稍候...",
}
