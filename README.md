
# TianGong Chat [![Docker Publish](https://github.com/linancn/TianGong-AI-Chat/actions/workflows/docker_publish.yml/badge.svg)](https://github.com/linancn/TianGong-AI-Chat/actions/workflows/docker_publish.yml)

## Env Preparing

### Using VSCode Dev Contariners

[Tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial)

Python 3 -> Additional Options -> 3.12-bullseye -> ZSH Plugins (Last One) -> Trust @devcontainers-contrib -> Keep Defaults

Setup `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt --upgrade

pip freeze > requirements_freeze.txt
```

```bash
sudo apt install python3.12-dev
```

## Start

```bash
export ui=tiangong-en

streamlit run Chat.py
```

### Auto Build

The auto build will be triggered by pushing any tag named like release-v$version. For instance, push a tag named as v0.0.1 will build a docker image of 0.0.1 version.

```bash
#list existing tags
git tag
#creat a new tag
git tag v0.0.1
#push this tag to origin
git push origin v0.0.1
```

### Production Run

```bash
docker build -t 339712838008.dkr.ecr.us-east-1.amazonaws.com/tiangong-chat:0.0.1 .

aws ecr get-login-password --region us-east-1  | docker login --username AWS --password-stdin 339712838008.dkr.ecr.us-east-1.amazonaws.com

docker push 339712838008.dkr.ecr.us-east-1.amazonaws.com/tiangong-chat:0.0.1

docker run -d -p 80:8501 -e ui=tiangong-en 339712838008.dkr.ecr.us-east-1.amazonaws.com/tiangong-chat:0.0.1
```
