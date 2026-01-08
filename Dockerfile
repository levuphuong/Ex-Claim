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
    xlsxwriter \
    fastapi \
    marisa-trie \
    huggingface_hub

# fairseq OK khi pip <24.1
RUN pip install fairseq==0.12.2

# (nếu dùng GENRE)
RUN git clone https://github.com/facebookresearch/GENRE.git
RUN pip install -e GENRE


# ===========================
# ENV + WORKDIR
# ===========================
ENV MODEL_DIR="outputs/Pretrained-models/mGENRE" \
    MODEL_NAME="fairseq_multilingual_entity_disambiguation"

# ===========================
# Tạo folder model
# ===========================
RUN mkdir -p ${MODEL_DIR}

WORKDIR /app/${MODEL_DIR}

# ===========================
# Download + Extract mGENRE
# ===========================
RUN if [ ! -d "${MODEL_NAME}" ]; then \
        echo "⬇️ Downloading mGENRE fairseq model..."; \
        wget -c https://dl.fbaipublicfiles.com/GENRE/${MODEL_NAME}.tar.gz; \
        tar --no-same-owner -xvf ${MODEL_NAME}.tar.gz; \
        rm ${MODEL_NAME}.tar.gz; \
    else \
        echo "✅ Model already exists: ${MODEL_NAME}"; \
    fi


# Download trie (BẮT BUỘC)
RUN hf download facebook/mgenre-wiki \
      titles_lang_all105_marisa_trie_with_redirect.pkl \
      --local-dir ${MODEL_DIR}

# Optional nhưng nên có
RUN hf download facebook/mgenre-wiki \
      config.json \
      --local-dir ${MODEL_DIR}


# Optional log
RUN echo "===== mGENRE download completed ====="


COPY . .

# Cho phép chạy script
RUN chmod +x run.sh

# HF bắt buộc có CMD
CMD ["bash", "run.sh"]
