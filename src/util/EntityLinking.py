import pickle
import numpy as np
import torch
from genre.trie import Trie, MarisaTrie
from genre.fairseq_model import mGENRE
import logging


def getModel(path):
    """
    Load entity linking model
    :param path: project path
    :return: model and trie
    """
    model_path = path + "Pretrained-models/mGENRE/"
    with open(path + "Pretrained-models/mGENRE/" + "titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
        trie = pickle.load(f)
    logging.info("Entity linking prefix tree loaded")

    model = mGENRE.from_pretrained(model_path + "fairseq_multilingual_entity_disambiguation").eval()
    logging.info("Entity linking model loaded")

    return model, trie


def getWikiEntities(path, ner_tagged_sentences):
    """
    Get wikipedia entities tagged in the sentences
    :param path: project path
    :param ner_tagged_sentences: ner tagged sentences
    :return: wikipedia entity names and scores
    """
    model, trie = getModel(path)
    wiki_entities = {}

    logging.info("Started extracting wiki entities")

    for language in ner_tagged_sentences:
        wiki_entities[language] = {}

        for data_type in ner_tagged_sentences[language]:
            data = ner_tagged_sentences[language][data_type]
            indexes = data["indexes"]
            sentences = data["sentences"]
            sentence_ids = data["sentence_ids"]

            results = model.sample(sentences,
                                   beam=1,
                                   prefix_allowed_tokens_fn=lambda batch_id, sent: [
                                        e for e in trie.get(sent.tolist())
                                        if e < len(model.task.target_dictionary)
                                    ],
                                   )
            scores = [x[0]["score"].item() for x in results]
            entities = [x[0]["text"].split(" >> ")[0] for x in results]
            languages = [x[0]["text"].split(" >> ")[1] for x in results]

            # Compute number of wiki entities
            scores_np = np.array(scores)
            wiki_mask = scores_np > -0.15
            no_of_wiki_entities = np.sum(wiki_mask)
            logging.info(f"Number of wiki entities linked for threshold -0.15 in language {language} "
                         f"- {no_of_wiki_entities}")

            wiki_entities[language][data_type] = {}
            wiki_entities[language][data_type]["scores"] = scores
            wiki_entities[language][data_type]["entity"] = entities
            wiki_entities[language][data_type]["languages"] = languages
            wiki_entities[language][data_type]["indexes"] = indexes
            wiki_entities[language][data_type]["sentence_ids"] = sentence_ids

    logging.info("Completed extracting wiki entities")

    return wiki_entities


def getWikiEntityPresenceScore(path, ner_tagged_sentences, ner_vectors):
    """
    Get vector representation of log probabilities indicating whether the entity is a wiki entity
    :param path: project path
    :param ner_tagged_sentences: ner tagged sentences
    :param ner_vectors: ner vectors
    :return: vector with log probabilities
    """
    wiki_entities = getWikiEntities(path, ner_tagged_sentences)
    wiki_entity_scores = {}

    logging.info("Started extracting entity presence scores")

    for language in wiki_entities:
        wiki_entity_scores[language] = {}

        for data_type in wiki_entities[language]:
            ner = ner_vectors[language][data_type]
            dimensions = ner.size()
            number_of_sentences = dimensions[0]
            max_length = dimensions[1]
            e_scores = torch.full((number_of_sentences, max_length, 1), -10.0)

            scores = wiki_entities[language][data_type]["scores"]
            indexes = wiki_entities[language][data_type]["indexes"]
            sentence_ids = wiki_entities[language][data_type]["sentence_ids"]

            for i in range(len(sentence_ids)):
                s_id = sentence_ids[i]
                t_ids = indexes[i]
                e_score = scores[i]

                for t_id in t_ids:
                    e_scores[s_id][t_id][0] = e_score

            wiki_entity_scores[language][data_type] = e_scores

    logging.info("Completed extracting entity presence scores")

    return wiki_entity_scores


def getWikiEntityPresenceVector(wiki_entity_scores, threshold):

    wiki_entity_vector = {}

    for language in wiki_entity_scores:
        wiki_entity_vector[language] = {}

        for data_type in wiki_entity_scores[language]:
            entity_scores = wiki_entity_scores[language][data_type]
            entity_vectors = torch.where(entity_scores > threshold, torch.tensor(1), torch.tensor(0))
            logging.info(f"Total number of wiki entity tokens present : {entity_vectors.sum()}")
            wiki_entity_vector[language][data_type] = entity_vectors

    logging.info("Wiki entity presence vector created")
    return wiki_entity_vector


def getELIndexVector(index_vectors, wiki_entity_scores, threshold, no_entity):
    """
    Incorporate entity linking details to index vector
    :param index_vectors: index vector indicating NER type
    :param el_vectors: entity linking vectors indicating their popularity
    :return: index vector with entity linking details, no of entity types
    """
    el_vectors = getWikiEntityPresenceVector(wiki_entity_scores, threshold)
    el_index_vectors = {}

    for language in index_vectors:
        el_index_vectors[language] = {}

        for data_type in index_vectors[language]:
            i_vector = index_vectors[language][data_type]
            el_vector = el_vectors[language][data_type].squeeze(2)
            el_index = i_vector + (el_vector * (no_entity - 1))
            el_index_vectors[language][data_type] = el_index

    no_entity = int(((no_entity - 1) * 2) + 1)
    logging.info(f"Number of entity types: {no_entity}")

    return el_index_vectors, no_entity