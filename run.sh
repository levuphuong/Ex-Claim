#!/bin/bash

echo "STEP 1: create directory"
mkdir -p outputs/Data/CheckThat-2022/english/verifiable-claim-detection

echo "===== DOWNLOAD & PREPARE CHECKTHAT 2022 DATASETS ====="

# =========================
# Language configurations
# =========================
declare -A LANG_URLS

LANG_URLS=(
  ["english"]="https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task1/data/subtasks-english/CT22_english_1A_checkworthy.zip?inline=false"
  ["dutch"]="https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task1/data/subtasks-dutch/CT22_dutch_1A_checkworthy.zip?inline=false"
)


TEST_URLS=(
  ["english"]="https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task1/data/subtasks-english/CT22_english_1B_claim.zip?inline=false"
  ["dutch"]="https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task1/data/subtasks-dutch/test/CT22_dutch_1B_claim_test.zip?inline=false"
)

BASE_OUTPUT="outputs/Data/CheckThat-2022"

# =========================
# Loop over languages
# =========================
for LANG in "${!LANG_URLS[@]}"; do
  echo "======================================"
  echo "Processing language: $LANG"
  echo "======================================"

  ZIP_FILE="CT22_${LANG}_1A.zip"
  TMP_DIR="ct22_${LANG}_tmp"
  OUT_DIR="$BASE_OUTPUT/$LANG/verifiable-claim-detection"

  # -------------------------
  # Train / Dev
  # -------------------------
  echo "STEP 1: download train/dev dataset"
  wget -q -O "$ZIP_FILE" "${LANG_URLS[$LANG]}"

  echo "STEP 2: unzip train/dev"
  rm -rf "$TMP_DIR"
  mkdir -p "$TMP_DIR"
  unzip -q "$ZIP_FILE" -d "$TMP_DIR"

  echo "STEP 3: locate files"
  TRAIN_FILE=$(find "$TMP_DIR" -iname "*train.tsv" | head -n 1)
  DEV_FILE=$(find "$TMP_DIR" -iname "*dev.tsv" | head -n 1)
  DEV_TEST_FILE=$(find "$TMP_DIR" -iname "*dev_test.tsv" | head -n 1)

  echo "  train = $TRAIN_FILE"
  echo "  dev   = $DEV_FILE"

  echo "STEP 4: copy train/dev"
  mkdir -p "$OUT_DIR"
  cp "$TRAIN_FILE" "$OUT_DIR/train.tsv"
  cp "$DEV_FILE"   "$OUT_DIR/dev.tsv"
  cp "$DEV_FILE"   "$OUT_DIR/dev_test.tsv"

  # -------------------------
  # Test (nếu có URL riêng)
  # -------------------------
  if [[ -n "${TEST_URLS[$LANG]}" ]]; then
    echo "STEP 5: download test dataset"
    TEST_ZIP="CT22_${LANG}_test.zip"
    TEST_TMP="${TMP_DIR}_test"

    wget -q -O "$TEST_ZIP" "${TEST_URLS[$LANG]}"

    echo "STEP 6: unzip test"
    rm -rf "$TEST_TMP"
    mkdir -p "$TEST_TMP"
    unzip -q "$TEST_ZIP" -d "$TEST_TMP"

    TEST_FILE=$(find "$TEST_TMP" -iname "*test*.tsv" | head -n 1)

    echo "  test  = $TEST_FILE"

    cp "$TEST_FILE" "$OUT_DIR/test.tsv"
  else
    echo "STEP 5: no official test → fallback to dev"
    cp "$OUT_DIR/dev.tsv" "$OUT_DIR/dev_test.tsv"
  fi

  echo "DONE: $LANG"
  echo
done

echo "===== ALL LANGUAGES PROCESSED ====="

echo "===== Downloading mGENRE pretrained model ====="

MODEL_DIR="outputs/Pretrained-models/mGENRE"
mkdir -p ${MODEL_DIR}

# Cài CLI mới
pip install --quiet huggingface_hub

# Download trie (BẮT BUỘC)
hf download facebook/mgenre-wiki \
  titles_lang_all105_marisa_trie_with_redirect.pkl \
  --local-dir ${MODEL_DIR}

echo "✓ Downloaded titles_lang_all105_marisa_trie_with_redirect.pkl"

# (Optional nhưng nên có)
hf download facebook/mgenre-wiki \
  config.json \
  --local-dir ${MODEL_DIR}

echo "✓ Downloaded config.json"

echo "===== mGENRE download completed ====="
ls -lh ${MODEL_DIR}


# echo "STEP 0: install missing python packages (debug)"
# pip install --no-cache-dir pandas==2.1.4

# echo "===== INSTALL RUNTIME DEPENDENCIES ====="

# pip install --no-cache-dir \
#     beautifulsoup4 \
#     pandas \
#     gensim \
#     tensorboardX \
#     sacremoses \
#     sentencepiece \
#     scikit-learn \
#     "scipy<1.12" \
#     "numpy<2.0"  \
#     "xlsxwriter"


echo "===== DEPENDENCIES INSTALLED ====="


##############
export PYTHONPATH=$PWD

mkdir -p outputs/Logs
mkdir -p outputs/Models/Claim-Detection
mkdir -p outputs/Data/All
mkdir -p outputs/Results/Claim-Detection
python src/evaluation/claim-detectionE.py
