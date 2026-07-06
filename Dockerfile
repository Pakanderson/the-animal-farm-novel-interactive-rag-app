FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/
COPY vector_index/ ./vector_index/

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 7860

# Fixed: Using double quotes eliminates the /bin/sh syntax execution error!
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]