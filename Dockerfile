# app/Dockerfile

FROM python:3.11.9-slim

WORKDIR /thai_mace

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://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@github.com/wachiranun/Thai-MACE.git .

RUN pip3 install -r requirement.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=127.0.0.1"]
