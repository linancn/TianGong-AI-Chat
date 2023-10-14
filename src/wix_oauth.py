import base64
import hashlib
import json

import requests
import streamlit as st

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


def get_orders(member_access_token: str):
    orders_response = requests.get(
        "https://www.wixapis.com/pricing-plans/v2/member/orders",
        headers={"authorization": member_access_token},
    )
    orders = json.loads(orders_response.text)

    return orders
