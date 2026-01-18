import os
import torch
import logging
import src.util.Util as Util

# === TÍCH HỢP PHONLP ===
try:
    import phonlp
    PHONLP_AVAILABLE = True
except ImportError:
    PHONLP_AVAILABLE = False
# ======================

PRETRAINED_MODEL_DIR = "Pretrained-models/"
logger = logging.getLogger("Cross-lingual-claim-detection")

# Global Cache cho PhoNLP
_PHONLP_MODEL = None

# THỨ TỰ ID CHUẨN (Khớp 100% với log WikiNeural của bạn)
PHONLP_TAGS_MAP = {
    'O': 0,
    'B-PER': 1, 'I-PER': 2,
    'B-ORG': 3, 'I-ORG': 4,
    'B-LOC': 5, 'I-LOC': 6,
    'B-MISC': 7, 'I-MISC': 8
}

def loadPhoNLP(save_dir="models/phonlp"):
    global _PHONLP_MODEL
    if _PHONLP_MODEL is None:
        if not os.path.exists(save_dir):
            logger.info(f"Downloading PhoNLP to {save_dir}...")
            os.makedirs(save_dir, exist_ok=True)
            phonlp.download(save_dir=save_dir)
        
        logger.info(f"Loading PhoNLP model from {save_dir}...")
        _PHONLP_MODEL = phonlp.load(save_dir=save_dir)
    return _PHONLP_MODEL

def getPhoNLPLabelsWithOffsets(text, model):
    """
    Chạy PhoNLP và trả về format nhãn kèm vị trí ký tự (start, end)
    """
    try:
        # PhoNLP annotate trả về list of lists (do có thể nhận nhiều câu)
        annotations = model.annotate(text=text)
        words = annotations[0][0]
        tags = annotations[2][0]
    except Exception as e:
        logger.error(f"PhoNLP error processing: {str(e)}")
        return []

    ner_labels = []
    cursor = 0 

    for word, tag in zip(words, tags):
        # PhoNLP dùng "_" nối từ ghép (VD: Hà_Nội), text gốc dùng khoảng trắng
        search_word = word.replace("_", " ")
        start_index = text.find(search_word, cursor)
        
        if start_index != -1:
            end_index = start_index + len(search_word)
            if tag != 'O':
                # Chuẩn hóa tag trơn thành loại thực thể (VD: B-LOC -> LOC)
                entity_type = tag.replace("B-", "").replace("I-", "")
                ner_labels.append({
                    "entity_type": entity_type,
                    "start": start_index,
                    "end": end_index,
                    "word": search_word
                })
            cursor = end_index
        else:
            cursor += len(search_word)

    return ner_labels

def getPhoNERVectors(path, model_name, dataset, tokenized_input):
    """
    Hàm chính điều phối việc trích xuất NER bằng PhoNLP
    """
    if not PHONLP_AVAILABLE:
        raise ImportError("PhoNLP is not installed. Please install it using: pip install phonlp")

    model = loadPhoNLP()
    NER_TAGS = PHONLP_TAGS_MAP
    ner_vectors = {}
    ner_tagged_dataset = {}

    logger.info("PhoNLP NER Vector creation started (Optimized Mapping)")

    for language in dataset:
        ner_vectors[language] = {}
        ner_tagged_dataset[language] = {}

        for data_type in dataset[language]:
            df = dataset[language][data_type]
            X = list(df['preprocessed_text'])
            
            # 1. Trích xuất nhãn từ PhoNLP
            all_ner_labels = []
            for text_sample in X:
                labels = getPhoNLPLabelsWithOffsets(text_sample, model)
                all_ner_labels.append(labels)
            
            # 2. Map nhãn vào vector của Subword (XLM-R/BERT)
            tokens = tokenized_input[language][data_type]
            offsets = tokens['offset_mapping']

            vectors, updated_ner_labels = computeNERVectors(NER_TAGS, all_ner_labels, offsets)
            ner_vectors[language][data_type] = vectors

            # 3. Tạo dữ liệu câu đã đánh dấu [START]/[END]
            ner_tagged_dataset[language][data_type] = {}
            # Format lại label để hàm tag hiểu (thêm prefix B-)
            for sent_idx in range(len(updated_ner_labels)):
                for label_idx in range(len(updated_ner_labels[sent_idx])):
                    updated_ner_labels[sent_idx][label_idx]['entity'] = "B-" + updated_ner_labels[sent_idx][label_idx]['entity_type']

            indexes, sentences, sentence_ids = getNERTaggedSentences(X, updated_ner_labels)
            ner_tagged_dataset[language][data_type]["indexes"] = indexes
            ner_tagged_dataset[language][data_type]["sentences"] = sentences
            ner_tagged_dataset[language][data_type]["sentence_ids"] = sentence_ids

            stats = torch.sum(vectors, dim=(0, 1)).tolist()
            logger.info(f"Entities stats for {language}-{data_type}: {stats}")
            
    logger.info("NER Vector creation completed")
    Util.clearMemory()
    return ner_vectors, ner_tagged_dataset

def computeNERVectors(NER_TAGS, NER_labels_list, offsets_list):
    """
    Logic Mapping Sửa đổi: Đảm bảo subwords nhận đúng nhãn thực thể từ từ ghép
    """
    number_of_sentences = len(offsets_list)
    max_length = len(offsets_list[0])
    ner_vec_length = len(NER_TAGS.keys())

    # Khởi tạo vector với nhãn 'O' (index 0) là 1
    ner_vectors = torch.zeros(number_of_sentences, max_length, ner_vec_length)
    ner_vectors[:, :, 0] = 1 

    for sentence_id in range(number_of_sentences):
        sentence_entities = NER_labels_list[sentence_id]
        sentence_offsets = offsets_list[sentence_id]

        for entity in sentence_entities:
            e_type = entity['entity_type']
            e_start = entity['start']
            e_end = entity['end']
            
            is_first_subword = True
            
            for token_id in range(max_length):
                t_start, t_end = sentence_offsets[token_id]
                
                # Bỏ qua special tokens của BERT/XLM-R
                if t_start == 0 and t_end == 0:
                    continue

                # KIỂM TRA BOUNDARY: Nếu subword nằm trong phạm vi thực thể
                if t_start >= e_start and t_end <= e_end:
                    prefix = "B-" if is_first_subword else "I-"
                    tag_name = prefix + e_type
                    
                    if tag_name in NER_TAGS:
                        tag_id = NER_TAGS[tag_name]
                        ner_vectors[sentence_id][token_id][0] = 0 # Tắt nhãn O
                        ner_vectors[sentence_id][token_id][tag_id] = 1
                        
                        if is_first_subword:
                            entity["token_id"] = token_id
                            is_first_subword = False

    return ner_vectors, NER_labels_list

def getPhoNERIndexVector(ner_vectors):
    """
    Chuyển đổi One-hot vector về Index vector (Case 2 trong logic của bạn)
    """
    index_vectors = {}
    no_entity = None

    for language in ner_vectors:
        index_vectors[language] = {}
        for data_type in ner_vectors[language]:
            vector = ner_vectors[language][data_type]

            if no_entity is None:
                no_entity = vector.shape[2]
            
            # Lấy index có giá trị lớn nhất (Argmax)
            i_vector = torch.argmax(vector, dim=2) 
            # Gom nhóm B- và I- về cùng 1 ID thực thể chính
            i_vector = torch.ceil(i_vector/2).to(torch.int64) 
            index_vectors[language][data_type] = i_vector

    no_entity = int((no_entity - 1) / 2 + 1)
    logging.info(f"Number of entity types: {no_entity}")
    return index_vectors, no_entity

def getNERTaggedSentences(sentences, NER_labels):
    indexes = []
    tagged_sentences = []
    sentence_ids = []

    for sentence_id in range(len(sentences)):
        sentence = sentences[sentence_id]
        ner_tags = NER_labels[sentence_id]

        for ner_label in ner_tags:
            if "token_id" not in ner_label:
                continue

            tagged_sent = tagSentence(ner_label, sentence)
            tagged_sentences.append(tagged_sent)
            indexes.append([ner_label["token_id"]])
            sentence_ids.append(sentence_id)

    return indexes, tagged_sentences, sentence_ids

def tagSentence(entity_status, sentence):
    start = entity_status["start"]
    end = entity_status["end"]
    tagged_sent = sentence[:start] + "[START] " + sentence[start:end] + " [END]" + sentence[end:]
    return tagged_sent[0:1024]