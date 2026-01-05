FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 🔴 QUAN TRỌNG: downgrade pip
RUN pip install --upgrade "pip<24.1" setuptools wheel



# stack ổn định
RUN pip install \
    numpy==1.26.4 \
    scipy==1.10.1 \
    torch==2.1.2 \
    gensim==4.3.2 \
    spacy==3.7.2 \
    thinc==8.2.2 \
    transformers==4.41.2 \
    pandas==2.1.4   \
    beautifulsoup4 \
    tensorboardX \
    sacremoses \
    sentencepiece \
    scikit-learn \
    xlsxwriter

# fairseq OK khi pip <24.1
RUN pip install fairseq==0.12.2

# (nếu dùng GENRE)
RUN git clone https://github.com/facebookresearch/GENRE.git
RUN pip install -e GENRE

COPY . .

# Cho phép chạy script
RUN chmod +x run.sh

# HF bắt buộc có CMD
CMD ["bash", "run.sh"]
