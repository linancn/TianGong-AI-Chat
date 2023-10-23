import base64
import hashlib
import json

import requests
import streamlit as st
import streamlit.components.v1 as components

import ui_config

ui = ui_config.create_ui_from_config()

CLIENT_ID = st.secrets["wix_client_id"]


def generate_code_challenge(code_verifier):
    # SHA-256
    hashed_verifier = hashlib.sha256(code_verifier.encode()).digest()

    # Base64
    code_challenge = base64.urlsafe_b64encode(hashed_verifier).rstrip(b"=")
    return code_challenge.decode("utf-8")


def wix_get_access_token():
    request = requests.post(
        "https://www.wixapis.com/oauth2/token",
        headers={
            "content-type": "application/json",
        },
        json={"clientId": CLIENT_ID, "grantType": "anonymous"},
    )
    return request.json()["access_token"]


def wix_get_callback_url(access_token: str, username: str, password: str):
    login_url = f"https://www.wixapis.com/_api/iam/authentication/v2/login"
    redirect_url = f"https://www.wixapis.com/_api/redirects-api/v1/redirect-session"

    headers = {
        "authorization": access_token,
        "content-type": "application/json",
    }

    data = {"loginId": {"email": username}, "password": password}

    response = requests.post(login_url, headers=headers, json=data)

    try:
        response_text = response.json()
    except json.JSONDecodeError:
        return None

    if response_text.get("state") != "SUCCESS":
        return None

    session_token = response_text["sessionToken"]
    code_challenge = generate_code_challenge(st.session_state["code_verifier"])

    redirect_data = {
        "auth": {
            "authRequest": {
                "clientId": CLIENT_ID,
                "codeChallenge": code_challenge,
                "codeChallengeMethod": "S256",
                "responseMode": "web_message",
                "responseType": "code",
                "scope": "offline_access",
                "state": st.session_state["state"],
                "sessionToken": session_token,
            }
        }
    }

    redirect_response = requests.post(redirect_url, headers=headers, json=redirect_data)

    try:
        url_response = redirect_response.json()["redirectSession"]["fullUrl"]
    except (json.JSONDecodeError, KeyError):
        return None

    return url_response


def get_member_access_token(code: str):
    response = requests.post(
        "https://www.wixapis.com/oauth2/token",
        headers={"content-type": "content-type: application/json"},
        json={
            "client_id": CLIENT_ID,
            "grant_type": "authorization_code",
            "code": code,
            "codeVerifier": st.session_state["code_verifier"],
        },
    )
    member_access_token = response.json().get("access_token")

    return member_access_token


def get_highest_active_subscription(orders):
    # Define priority levels
    priority = {"Elite": 3, "Pro": 2, "Basic": 1}

    # Find all orders with "ACTIVE" status
    active_orders = [order for order in orders if order["status"] == "ACTIVE"]

    if not active_orders:
        return None

    # Get the order with the highest level
    highest_order = max(active_orders, key=lambda x: priority.get(x["planName"], 0))

    return highest_order["planName"]


def get_subscription(member_access_token: str) -> str:
    orders_response = requests.get(
        "https://www.wixapis.com/pricing-plans/v2/member/orders",
        headers={"authorization": member_access_token},
    )
    orders = json.loads(orders_response.text)["orders"]

    subscription = get_highest_active_subscription(orders)

    return subscription


def check_wix_oauth() -> (bool, str, str):
    component_url = st.secrets["component_url"]
    placeholder = st.empty()

    with placeholder.container():
        _, col_center, _ = st.columns(3)

        with col_center:
            st.markdown(ui.wix_login_title, unsafe_allow_html=True)
            with st.form(key="login_form"):
                username = st.text_input(ui.wix_login_username_label)
                password = st.text_input(ui.wix_login_password_label, type="password")
                submit = st.form_submit_button(
                    ui.wix_login_button_label, type="primary", use_container_width=True
                )
            st.link_button(
                label=ui.wix_signup_button_label,
                url=ui.wix_signup_button_url,
                use_container_width=True,
            )

        if submit:
            wix_access_token = wix_get_access_token()
            st.session_state["wix_callback_url"] = wix_get_callback_url(
                access_token=wix_access_token, username=username, password=password
            )

        if "wix_callback_url" in st.session_state:
            if st.session_state["wix_callback_url"] is not None:
                wix_component = components.declare_component(
                    "wix_component", url=component_url
                )
                wix_component_post = components.declare_component(
                    "wix_component_post", url=component_url
                )

                if "wix_first_run" not in st.session_state:
                    st.session_state["wix_return_data"] = wix_component(
                        url=st.session_state["wix_callback_url"]
                    )
                    st.session_state["wix_first_run"] = True
                # At least run twice, with the state code unchanged to get the code to login in
                if st.session_state["wix_return_data"] is None:
                    st.session_state["wix_return_data"] = wix_component_post(
                        url=st.session_state["wix_callback_url"]
                    )
                else:
                    member_access_token = get_member_access_token(
                        code=st.session_state["wix_return_data"]
                    )
                    subsription = get_subscription(
                        member_access_token=member_access_token
                    )

                    auth = True

                    placeholder.empty()

                    return auth, username, subsription
            else:
                with col_center:
                    st.error(ui.wix_login_error_text, icon=ui.wix_login_error_icon)
                return False, None, None
        else:
            return False, None, None
