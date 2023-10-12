import base64
import hashlib
import json
import uuid

import requests
import streamlit as st
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="your-secret-key")


# 你的配置信息
CLIENT_ID = st.secrets["wix_client_id"]
REDIRECT_URI = st.secrets["wix_redirect_uri"]


def get_access_token():
    request = requests.post(
        "https://www.wixapis.com/oauth2/token",
        headers={
            "content-type": "application/json",
        },
        json={"clientId": CLIENT_ID, "grantType": "anonymous"},
    )
    return request.json()["access_token"]


def generate_code_challenge(code_verifier):
    # 1. 使用SHA-256对code_verifier进行散列。
    hashed_verifier = hashlib.sha256(code_verifier.encode()).digest()

    # 2. 将散列值编码为Base64URL格式。
    code_challenge = base64.urlsafe_b64encode(hashed_verifier).rstrip(b"=")
    return code_challenge.decode("utf-8")


@app.get("/check_auth/")
def check_auth(request: Request = None):
    if (
        "plan_status" not in request.session
        or request.session["plan_status"] != "ACTIVE"
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")
    else:
        response = Response(content='{"status": "OK"}', media_type="application/json")
        response.headers["member_id"] = request.session["member_id"]
        response.headers["plan_name"] = request.session["plan_name"]
        return response


@app.get("/redirect_to_wix/")
def login(request: Request):
    # 生成state和code_verifier
    state = str(uuid.uuid4()).replace("-", "")
    code_verifier = str(uuid.uuid4()).replace("-", "")
    code_challenge = generate_code_challenge(code_verifier)
    request.session["state"] = state  # 存储state
    request.session["code_verifier"] = code_verifier  # 存储code_verifier

    # 构造请求参数
    auth_request = {
        "redirectUri": REDIRECT_URI,
        "clientId": CLIENT_ID,
        "codeChallenge": code_challenge,
        "codeChallengeMethod": "S256",
        "responseMode": "fragment",
        "responseType": "code",
        "scope": "offline_access",
        "state": state,
    }

    ACCESS_TOKEN = get_access_token()
    # 请求登录URL
    response = requests.post(
        "https://www.wixapis.com/_api/redirects-api/v1/redirect-session",
        headers={"authorization": ACCESS_TOKEN, "content-type": "application/json"},
        json={"auth": {"authRequest": auth_request}},
    )
    data = response.json()
    login_url = data["redirectSession"]["fullUrl"]

    # 重定向到登录页面
    return RedirectResponse(login_url)


templates = Jinja2Templates(directory="src/templates")


@app.get("/fragment", response_class=HTMLResponse)
async def get_fragment_page(request: Request):
    return templates.TemplateResponse("fragment.html", {"request": request})


@app.get("/callback")
async def get_token(
    code: str = None,
    state: str = None,
    request: Request = None,
):
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    # 验证 state 以确保请求是从你的应用发起的
    saved_state = request.session.get("state")
    if not saved_state or saved_state != state:
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "code_verifier": request.session.get("code_verifier"),
    }

    response = requests.post(
        "https://www.wixapis.com/oauth2/token",
        data=token_data,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    token_response = response.json()
    access_token = token_response.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="Token request failed")

    request.session["access_token"] = access_token

    orders_response = requests.get(
        "https://www.wixapis.com/pricing-plans/v2/member/orders",
        headers={"authorization": access_token},
    )

    orders = json.loads(orders_response.text)

    request.session["plan_name"] = orders["orders"][0]["planName"]
    request.session["plan_status"] = orders["orders"][0]["status"]
    request.session["member_id"] = orders["orders"][0]["buyer"]["memberId"]

    response = RedirectResponse("https://test.tiangong.world/")
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
