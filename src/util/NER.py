import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import logging
import src.util.Util as Util

# === ADD ON TOP ===
try:
    from phonenlp import PhoNLP
    PHONLP_AVAILABLE = True
except:
    PHONLP_AVAILABLE = False
# ==================



PRETRAINED_MODEL_DIR = "Pretrained-models/"
logger = logging.getLogger("Cross-lingual-claim-detection")


def readPipeline(path, model_name):
    """
    Read pretrained ner model from local directory, if unavailable download and save to local directory
    :param path: project path
    :param model_name: pretrained ner model name
    :return: pipeline
    """
    MODEL_DIR = path + PRETRAINED_MODEL_DIR + model_name.replace("/", "-")
    logger.info(f"Loading NER model at {MODEL_DIR}")

    if os.path.exists(MODEL_DIR):
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        model, tokenizer = downloadPretrainedNERModel(path, model_name)

    device = Util.getDevice()
    model.to(device)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    NER_TAGS = model.config.label2id

    logger.info(f"NER Pipeline reading completed. NER Tags: {NER_TAGS}")
    return ner_pipeline, NER_TAGS


def downloadPretrainedNERModel(path, model_name):
    """
    Download pretrained ner model and tokenizer
    :param path: project path
    :param model_name: pretrained ner model name
    :return: pretrained ner model, tokenizer
    """
    MODEL_PATH = path + PRETRAINED_MODEL_DIR + model_name.replace("/", "-")
    logger.info(f"Downloading NER model at {MODEL_PATH}")

    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    os.makedirs(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    logger.info(f"Downloading completed")
    return model, tokenizer


def getNER(ner_pipeline, input_texts):
    """
    Ger NER labels for the input texts
    :param ner_pipeline:
    :param input_texts:
    :return:
    """
    NER_labels = ner_pipeline(input_texts)
    return NER_labels


def getNERVectors(path, model_name, dataset, tokenized_input):
    """
    Get NER vector representation of the dataset, and NER tagged sentences
    :param path: project path
    :param model_name: NER model name
    :param dataset: dataset
    :param tokenized_input: tokenized input
    :return: NER vector representation of the dataset, and NER tagged sentences
    """
    ner_pipeline, NER_TAGS = readPipeline(path, model_name)
    ner_vectors = {}
    ner_tagged_dataset = {}

    logger.info("NER Vector creation started")

    for language in dataset:
        ner_vectors[language] = {}
        ner_tagged_dataset[language] = {}

        for data_type in dataset[language]:
            df = dataset[language][data_type]
            X = list(df['preprocessed_text'])
            NER_labels = getNER(ner_pipeline, X)

            tokens = tokenized_input[language][data_type]
            offsets = tokens['offset_mapping']

            # indexes indicate list of indexes of NER positions in tokenized data
            vectors, NER_labels = computeNERVectors(NER_TAGS, NER_labels, offsets)
            ner_vectors[language][data_type] = vectors

            ner_tagged_dataset[language][data_type] = {}
            indexes, sentences, sentence_ids = getNERTaggedSentences(X, NER_labels)
            ner_tagged_dataset[language][data_type]["indexes"] = indexes
            ner_tagged_dataset[language][data_type]["sentences"] = sentences
            ner_tagged_dataset[language][data_type]["sentence_ids"] = sentence_ids

            stats = torch.sum(vectors, dim=(0, 1)).tolist()
            logger.info(f"Entities statistics for the language {language} and data {data_type} : {stats}")
    logger.info("NER Vector creation completed")

    del ner_pipeline
    Util.clearMemory()
    return ner_vectors, ner_tagged_dataset


def computeNERVectors(NER_TAGS, NER_labels, offsets):
    """
    Compute NER vectors using NER labels and token offsets
    :param NER_TAGS: list of all the NER tags
    :param NER_labels: NER labels of the tokens
    :param offsets: offset of the tokens
    :return: NER vectors of the tokens, updated NER_labels with token ids
    """

    number_of_sentences = len(offsets)
    max_length = len(offsets[0])
    ner_vec_length = len(NER_TAGS.keys())

    default_ner_vector = torch.zeros(ner_vec_length)
    default_ner_vector[0] = 1

    ner_vectors = default_ner_vector.unsqueeze(0).repeat(number_of_sentences, max_length, 1)

    for sentence_id in range(number_of_sentences):
        sentence_tags = NER_labels[sentence_id]
        sentence_offsets = offsets[sentence_id]

        starting_token_id = 0

        for ner_label in sentence_tags:
            entity = ner_label['entity']
            entity_id = NER_TAGS[entity]
            start = ner_label['start']
            end = ner_label['end']

            for token_id in range(starting_token_id, max_length):
                token_start = sentence_offsets[token_id][0]
                token_end = sentence_offsets[token_id][1]

                if token_start >= start and end <= token_end:
                    ner_vectors[sentence_id][token_id][0] = 0
                    ner_vectors[sentence_id][token_id][entity_id] = 1
                    starting_token_id = token_id + 1
                    ner_label["token_id"] = token_id  # Insert token details to the NER labels

                    break

    return ner_vectors, NER_labels


def getIndexVector(ner_vectors):
    """
    Create an NER type index vector
    :param ner_vectors: one-hot vector representation of NER
    :return: index vector, no of entity types
    """
    index_vectors = {}
    no_entity = None

    for language in ner_vectors:
        index_vectors[language] = {}

        for data_type in ner_vectors[language]:
            vector = ner_vectors[language][data_type]

            if no_entity is None:
                no_entity = vector.shape[2]
            i_vector = torch.argmax(vector, dim=2)  # Find entity index
            i_vector = torch.ceil(i_vector/2).to(torch.int64)  # Bring start and body indication to single entity index
            index_vectors[language][data_type] = i_vector

    no_entity = int((no_entity - 1) / 2 + 1)
    logging.info(f"Number of entity types: {no_entity}")
    return index_vectors, no_entity


def getNERTaggedSentences(sentences, NER_labels):
    """
    Get NER tagged with [START] and [END] in each sentence for each NER
    :param sentences: Input sentences
    :param NER_labels: NER details generated by the pipeline
    :return: indexes indicating position of the NER in tokenized data, NER tagged sentences, sentence ids in the dataset
    """

    indexes = []
    tagged_sentences = []
    sentence_ids = []

    number_of_sentences = len(sentences)

    for sentence_id in range(number_of_sentences):
        sentence = sentences[sentence_id]
        ner_tags = NER_labels[sentence_id]

        complete_entity_status = None

        for ner_label in ner_tags:  # Processed at detailed entity tag (start and end of entities are tagged)
            # start and end index of the entity in text
            if "token_id" not in ner_label:  # If the entity is not available in the tokenized data
                continue

            entity = ner_label['entity']
            start = ner_label['start']
            end = ner_label['end']
            token_id = ner_label["token_id"]

            if "B-" in entity:  # Beginning of an entity. Create entity status. Save if there is a previous entity
                if complete_entity_status is not None:  # Already an entity existing
                    tagged_sent = tagSentence(complete_entity_status, sentence)
                    tagged_sentences.append(tagged_sent)
                    indexes.append(complete_entity_status["token_id"])
                    sentence_ids.append(complete_entity_status["sentence_id"])

                complete_entity_status = {"start": start, "end": end, "token_id": [token_id],
                                          "sentence_id": sentence_id}
            elif "I-" in entity:  # Middle of an entity. Update entity status if existing
                if complete_entity_status is not None:
                    complete_entity_status["end"] = end
                    complete_entity_status["token_id"].append(token_id)
            else:
                if complete_entity_status is not None:  # Save if there is a previous entity
                    tagged_sent = tagSentence(complete_entity_status, sentence)
                    tagged_sentences.append(tagged_sent)
                    indexes.append(complete_entity_status["token_id"])
                    sentence_ids.append(complete_entity_status["sentence_id"])
                    complete_entity_status = None

        if complete_entity_status is not None:  # Save if there is a previous entity
            tagged_sent = tagSentence(complete_entity_status, sentence)
            tagged_sentences.append(tagged_sent)
            indexes.append(complete_entity_status["token_id"])
            sentence_ids.append(complete_entity_status["sentence_id"])

    logger.info(f"NER tagged sentences created. Size - {len(tagged_sentences)}")
    return indexes, tagged_sentences, sentence_ids


def tagSentence(complete_entity_status, sentence):
    """
    Tag an entity in a sentence with [START] and [END]
    :param complete_entity_status: status of the entity
    :param sentence: sentence to be tagged
    :return: entity tagged sentence
    """
    start = complete_entity_status["start"]
    end = complete_entity_status["end"]
    tagged_sent = sentence[:start] + "[START] " + sentence[start:end] + " [END]" + sentence[end:]
    return tagged_sent[0:1024]
