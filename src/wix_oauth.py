import base64
import hashlib
import json

import requests
import streamlit as st
import streamlit.components.v1 as components

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
    url = "https://www.wixapis.com/_api/iam/authentication/v2/login"

    headers = {
        "authorization": access_token,
        "content-type": "application/json",
    }

    data = {"loginId": {"email": username}, "password": password}

    response = requests.post(url, headers=headers, json=data)
    response_text = json.loads(response.text)

    code_challenge = generate_code_challenge(st.session_state["code_verifier"])

    if response_text["state"] == "SUCCESS":
        session_token = response_text["sessionToken"]
        respond = requests.post(
            "https://www.wixapis.com/_api/redirects-api/v1/redirect-session",
            headers={
                "authorization": access_token,
                "content-type": "application/json",
            },
            json={
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
            },
        )

        url_response = respond.json()["redirectSession"]["fullUrl"]

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


def check_wix_oauth():
    placeholder = st.empty()

    with placeholder.container():
        col_left, col_center, col_right = st.columns(3)

        with col_center:
            with st.form(key="login_form"):
                username = st.text_input("username")
                password = st.text_input("password", type="password")
                submit = st.form_submit_button("Login")

        if submit:
            wix_access_token = wix_get_access_token()
            st.session_state["wix_callback_url"] = wix_get_callback_url(
                access_token=wix_access_token, username=username, password=password
            )

        if "wix_callback_url" in st.session_state:
            wix_component = components.declare_component(
                "wix_component", url="https://test.tiangong.world/callback/"
            )
            wix_component_post = components.declare_component(
                "wix_component_post", url="https://test.tiangong.world/callback/"
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
                subsription = get_subscription(member_access_token=member_access_token)

                auth = True

                placeholder.empty()

                return auth, username, subsription
        else:
            return False, None, None
