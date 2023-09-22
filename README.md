
# TianGong Chat

## Env Preparing

### Using VSCode Dev Contariners

[Tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial)

Python 3 -> Additional Opentions -> 3.11-bullseye -> ZSH Plugins (Last One) -> Trust @devcontainers-contrib -> Keep Defaults

Setup `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt --upgrade
```

```bash
sudo apt install python3.11-dev
sudo apt install libmagic-dev
sudo apt install poppler-utils
sudo apt install tesseract-ocr
sudo apt install libreoffice
sudo apt install pandoc
```

Install Cuda (optional):

```bash
sudo apt install nvidia-cuda-toolkit
```

## Env Preparing in MacOS

Install `Python 3.11`

```bash
brew update
brew install python@3.11
```

Setup `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt --upgrade
```

```bash
brew update
brew install libmagic
brew install poppler-qt5
# echo 'export PATH="/usr/local/opt/poppler-qt5/bin:$PATH"' >> ~/.zshrc
# export LDFLAGS="-L/usr/local/opt/poppler-qt5/lib"
# export CPPFLAGS="-I/usr/local/opt/poppler-qt5/include"
brew install tesseract
# brew install tesseract-lang
# brew cleanup tesseract-lang
brew install libreoffice
brew install pandoc
```

## Start

```bash
export ui=tiangong-en

streamlit run Chat.py
```

### Auto Build

The auto build will be triggered by pushing any tag named like release-v$version. For instance, push a tag named as release-v0.0.1 will build a docker image of 0.0.1 version.

```bash
#list existing tags
git tag
#creat a new tag
git tag release-v0.0.1
#push this tag to origin
git push origin release-v0.0.1
```

### Production Run

```bash
docker run --detach \
    --name tiangong-chat \
    --restart=always \
    --expose 8501 \
    --net=tiangongbridge \
    --env ui=tiangong-en \
    --env VIRTUAL_HOST=YourURL \
    --env VIRTUAL_PORT=8501 \
    --env LETSENCRYPT_HOST=YourURL \
    --env LETSENCRYPT_EMAIL=YourEmail \
    image:tag

```

## To Do
