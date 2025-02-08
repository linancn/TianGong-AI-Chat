FROM --platform=linux/amd64 python:3.13-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container at /app/requirements.txt
COPY requirements_freeze.txt requirements.txt

# Upgrade pip
RUN pip install --upgrade pip

# Install pip packages
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY .streamlit/  ./.streamlit/
COPY src/ ./src/

# Command to run
CMD ["streamlit", "run", "src/Chat.py", "--server.port=8501", "--server.address=0.0.0.0"]
