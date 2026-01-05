import os
import torch
from transformers import AutoModel
from transformers import AutoTokenizer
import logging
import src.util.Util as Util

PRETRAINED_MODEL_DIR = "Pretrained-models/"
logger = logging.getLogger("Cross-lingual-claim-detection")


def readModelAndTokenizer(path, model_name):
    """
    Read pretrained model from local directory, if unavailable download and save to local directory
    :param path: project path
    :param model_name: pretrained model name
    :return: pretrained model, tokenizer
    """

    MODEL_DIR = path + PRETRAINED_MODEL_DIR + model_name.replace("/", "-")
    logger.info(f"Reading model and tokenizer at {MODEL_DIR}")

    if os.path.exists(MODEL_DIR):
        model = AutoModel.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        model, tokenizer = downloadPretrainedModel(path, model_name)

    device = Util.getDevice()
    model.eval()
    model.to(device)
    logger.info("Reading completed")
    return model, tokenizer


def padEmbeddings(word_embeddings, pad_input, padding_factor):
    """
    Pad embedding such that it is evenly divisible by padding factor
    :param word_embeddings: word embeddings
    :param pad_input: boolean indicating whether input should be padded
    :param padding_factor: division factor
    :return: padded embeddings
    """
    # No padding is required
    if not pad_input:
        return word_embeddings

    padded_embeddings = {}
    pad_size = None

    for language in word_embeddings:
        padded_embeddings[language] = {}

        for data_type in word_embeddings[language]:
            X = word_embeddings[language][data_type]

            if pad_size is None:
                current_size = X.shape[2]
                pad_size = (padding_factor - (current_size % padding_factor)) % padding_factor

                # No padding is required
                if pad_size == 0:
                    logger.info("Padding is not required")
                    return word_embeddings

            padded_input = torch.nn.functional.pad(X, (0, pad_size))
            padded_embeddings[language][data_type] = padded_input

    if pad_size is not None:
        logger.info(f"Padded size - {pad_size}")
    return padded_embeddings


def downloadPretrainedModel(path, model_name):
    """
    Download pretrained model and tokenizer
    :param path: project path
    :param model_name: pretrained model name
    :return: pretrained model, tokenizer
    """
    MODEL_PATH = path + PRETRAINED_MODEL_DIR + model_name.replace("/", "-")
    logger.info(f"Downloading model and tokenizer at {MODEL_PATH}")

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    os.makedirs(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    logger.info("Downloading completed")
    return model, tokenizer


def tokenizeDataset(tokenizer, dataset, max_length):
    """
    Tokenize and return Pytorch tensor dataset
    :param tokenizer: pretrained model tokenizer
    :param dataset: dataset in dictionary format e.g. {language: {train, dev, dec_test, test} }
    :param max_length: maximum sequence length
    :return: tokenized dataset in dictionary format
    """
    logger.info("Started tokenizing the dataset")

    encoded_X = {}
    encoded_Y = {}

    for language in dataset:
        encoded_X[language] = {}
        encoded_Y[language] = {}

        for data_type in dataset[language]:
            df = dataset[language][data_type]
            X = list(df['preprocessed_text'])
            labels = list(df['class_label'])

            X_tokenized = tokenizer(X, padding='max_length', truncation=True, max_length=max_length,
                                    return_tensors='pt', return_offsets_mapping=True)
            Y = torch.tensor(labels)
            encoded_X[language][data_type] = X_tokenized
            encoded_Y[language][data_type] = Y

    logger.info("Tokenization completed")
    return encoded_X, encoded_Y


def getEmbeddingRepresentation(model, x_tokenized):
    """
    Get embedding representation of list of tokenized input
    :param model: pretrained model
    :param x_tokenized: list of tokenized input
    :return: list of embedding vector
    """
    device = Util.getDevice()
    x = x_tokenized.copy()
    if 'offset_mapping' in x:
        del x['offset_mapping']
    x.to(device)
    with torch.no_grad():
        embeddings = model(**x)['last_hidden_state']

    return embeddings.cpu()


def generateEmbeddings(model, dataset):
    """
    Generate embedding representation of the dataset using the model
    :param model: pretrained model
    :param dataset: dataset in dictionary format
    :return: embedded dataset in dictionary format
    """
    word_embeddings = {}
    logger.info("Started generating embedding vectors")

    for language in dataset:
        word_embeddings[language] = {}

        for data_type in dataset[language]:
            X = dataset[language][data_type]
            X_embeddings = getEmbeddingRepresentation(model, X)
            word_embeddings[language][data_type] = X_embeddings

    logger.info("Completed generating embedding vectors")
    return word_embeddings


def getEmbeddedDataset(path, model_name, dataset, max_length):
    """
    Get embedding representation of the dataset for a given model
    :param path: project path
    :param model_name: model name
    :param dataset: dataset in dictionary format
    :param max_length: max length of the sequence
    :return: tokenized dataset, embedded_dataset
    """
    # Read pretrained model and tokenizer
    model, tokenizer = readModelAndTokenizer(path, model_name)

    # Tokenize and generate tensor dataset
    tokenized_input, class_labels = tokenizeDataset(tokenizer, dataset, max_length)
    word_embeddings = generateEmbeddings(model, tokenized_input)

    # Clear unused models
    del model
    del tokenizer
    Util.clearMemory()
    return tokenized_input, class_labels, word_embeddings


def concatenateEmbeddings(embedded_dataset, ner_vectors):
    """
    Concatenate 3D torch embeddings
    :param embedded_dataset: embedded dataset
    :param ner_vectors: ner vectors
    :return: concatenated embedding
    """
    device = Util.getDevice()
    concatenated_embeddings = {}
    for language in embedded_dataset:
        concatenated_embeddings[language] = {}

        for data_type in embedded_dataset[language]:
            e1 = embedded_dataset[language][data_type]
            e2 = ner_vectors[language][data_type]
            concatenated_embeddings[language][data_type] = torch.concat((e1, e2), dim=2)
    return concatenated_embeddings


def getMeanVector(embedded_dataset):
    pool_embeddings = {}
    for language in embedded_dataset:
        pool_embeddings[language] = {}

        for data_type in embedded_dataset[language]:
            e = embedded_dataset[language][data_type]
            pool_embeddings[language][data_type] = torch.mean(e, dim=2)
    return pool_embeddings


def getCLSVector(embedded_dataset):
    cls_embeddings = {}
    for language in embedded_dataset:
        cls_embeddings[language] = {}

        for data_type in embedded_dataset[language]:
            e = embedded_dataset[language][data_type]
            cls_embeddings[language][data_type] = e[:,0,:]
    return cls_embeddings
