
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

https://github.com/ollama/ollama

https://ollama.com/library/deepseek-r1
https://ollama.com/library/qwen2.5

```bash

ollama serve

# 4G
ollama run deepseek-r1:1.5b-qwen-distill-fp16
ollama run qwen2.5:1.5b-instruct-fp16

# 12G
ollama run deepseek-r1:14b-qwen-distill-q4_K_M
ollama run qwen2.5:14b-instruct

# 16G
ollama run deepseek-r1:7b-qwen-distill-fp16
ollama run qwen2.5:14b-instruct-q6_K

# 32G
ollama run deepseek-r1:14b-qwen-distill-fp16
ollama run qwen2.5:32b-instruct-q6_K

# 80G
ollama run deepseek-r1:70b-llama-distill-q8_0
ollama run qwen2.5:72b-instruct-q8_0
```

## Start

```bash
export ui=tiangong-en

streamlit run Chat.py
```

## GPU Monitoring

```bash
watch -n 1 nvidia-smi
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
