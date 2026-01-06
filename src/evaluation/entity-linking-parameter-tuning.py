import logging
import src.util.ReadDataset as ReadDataset
import src.util.Util as Util
import src.util.Embedding as Embedding
import src.util.Training as Training
import src.models.ClaimDetection as CD_Models
import src.util.Evaluation as Evaluation
import src.util.NER as NER
import src.util.EntityLinking as EntityLinking
from genre.trie import Trie, MarisaTrie
import numpy as np
import src.util.Results as Results

task = "verifiable-claim-detection"
run_name = "-PT-XLMR"
config = Util.readAllConfig()
path = config["path"]
claim_detection_config = config[task]
model_name = claim_detection_config["default-model"]
max_length = claim_detection_config["max-length"]  # Refer BertTweet Paper
training_languages = ["english", "dutch", "bulgarian", "turkish"]

# Read dataset
dataset = ReadDataset.readTrainTestData(path, task, False)

# Get embedding representation
tokenized_input, class_labels, word_embeddings = Embedding.getEmbeddedDataset(path, model_name, dataset, max_length)

# Common training parameters
batch_size = 32
hidden_size = 128
output_size = 2
learning_rate = 3e-5
epochs = 30
MODELS_PATH = path + "Models/Claim-Detection/"
iterations = 2
ee_size = 128
we_size = Training.getInputSize(word_embeddings)

########################################################################################################################

ner_model = "Babelscape/wikineural-multilingual-ner"
ner_vectors_cg, ner_tagged_sentences_cg = NER.getNERVectors(path, ner_model, dataset, tokenized_input)
entity_indexes, no_entity = NER.getIndexVector(ner_vectors_cg)

########################################################################################################################

# Wiki entity extraction using coarse-grained NER
wiki_entity_scores_cg = EntityLinking.getWikiEntityPresenceScore(path, ner_tagged_sentences_cg, ner_vectors_cg)

wiki_entity_presence_threshold_list = np.arange(-0.5, -0.05, 0.05)

wiki_entities_cg = None
ner_vectors_cg = None
ner_tagged_sentences_cg = None

for wiki_entity_presence_threshold in wiki_entity_presence_threshold_list:
    logging.info("Processing for threshold - ", wiki_entity_presence_threshold)
    entity_indexes, no_entity = EntityLinking.getELIndexVector(entity_indexes, wiki_entity_scores_cg,
                                                               wiki_entity_presence_threshold, no_entity)

    train_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels,
                                                        training_languages, ["train"], batch_size, True)
    validation_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels,
                                                             training_languages, ["dev"], batch_size, False)

    ###################################################################################################################
    # Word embedding + Wiki Entity Presence
    model_prefix = "EClassifier-NER-cWiki-" + str(int(wiki_entity_presence_threshold * 100)) + run_name
    logging.info(f"Started training {model_prefix}")

    for i in range(iterations):
        model_name = model_prefix + "-" + str(i) + ".pth"
        model = CD_Models.EXClaim(we_size, ee_size, no_entity, hidden_size, output_size)
        Training.trainModelWithEntityData(model, train_loader, validation_loader, learning_rate, epochs,
                                          MODELS_PATH + model_name)

        best_model = Training.loadModel(model, MODELS_PATH + model_name)
        Evaluation.evaluateTestDataWithEntity(best_model, model_name, word_embeddings, entity_indexes, dataset,
                                              "dev_test", path)
    logging.info(f"Completed training {model_prefix}")

    input_vectors = None
    wiki_entity_vectors_cg = None
    model = None
    best_model = None
    train_loader = None
    validation_loader = None
    Util.clearMemory()

wiki_entity_scores_cg = None
Util.clearMemory()
Results.computeAveragePerformance(path, "dev_test") # No use remove
########################################################################################################################

ner_model = "multinerd-mbert"
ner_vectors_fg, ner_tagged_sentences_fg = NER.getNERVectors(path, ner_model, dataset, tokenized_input)
entity_indexes, no_entity = NER.getIndexVector(ner_vectors_fg)

########################################################################################################################

# Wiki entity extraction using fine-grained NER
wiki_entity_scores_fg = EntityLinking.getWikiEntityPresenceScore(path, ner_tagged_sentences_fg, ner_vectors_fg)

wiki_entity_presence_threshold_list = np.arange(-0.5, -0.05, 0.05)

wiki_entities_fg = None
ner_vectors_fg = None
ner_tagged_sentences_fg = None

for wiki_entity_presence_threshold in wiki_entity_presence_threshold_list:
    logging.info("Processing for threshold - ", wiki_entity_presence_threshold)
    entity_indexes, no_entity = EntityLinking.getELIndexVector(entity_indexes, wiki_entity_scores_fg,
                                                               wiki_entity_presence_threshold, no_entity)

    train_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels,
                                                        training_languages, ["train"], batch_size,
                                                        True)
    validation_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels,
                                                             training_languages, ["dev"], batch_size,
                                                             False)
    ####################################################################################################################
    # Word embedding + Wiki Entity Presence
    model_prefix = "EClassifier-NER-fWiki-" + str(int(wiki_entity_presence_threshold * 100)) + run_name
    logging.info(f"Started training {model_prefix}")

    for i in range(iterations):
        model_name = model_prefix + "-" + str(i) + ".pth"
        model = CD_Models.EXClaim(we_size, ee_size, no_entity, hidden_size, output_size)
        Training.trainModelWithEntityData(model, train_loader, validation_loader, learning_rate, epochs,
                                          MODELS_PATH + model_name)

        best_model = Training.loadModel(model, MODELS_PATH + model_name)
        Evaluation.evaluateTestDataWithEntity(best_model, model_name, word_embeddings, entity_indexes, dataset,
                                              "dev_test", path)
    logging.info(f"Completed training {model_prefix}")

    input_vectors = None
    wiki_entity_vectors_fg = None
    model = None
    best_model = None
    train_loader = None
    validation_loader = None
    Util.clearMemory()
Results.computeAveragePerformance(path, "dev_test")