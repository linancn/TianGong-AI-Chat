from fastapi import FastAPI, Depends, HTTPException, Response, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

app = FastAPI()

security = HTTPBasic()

VALID_USERNAME = "admin@admin.com"
VALID_PASSWORD = "password"


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, VALID_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, VALID_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials


@app.get("/redirect_to_streamlit/")
def redirect_to_streamlit(
    credentials: HTTPBasicCredentials = Depends(verify_credentials),
):
    username = credentials.username

    response = Response(content='{"status": "OK"}', media_type="application/json")
    response.headers["username"] = username

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
