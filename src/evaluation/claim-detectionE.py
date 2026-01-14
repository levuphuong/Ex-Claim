import logging
import src.util.ReadDataset as ReadDataset
import src.util.Util as Util
import src.util.Embedding as Embedding
import src.util.Training as Training
import src.models.ClaimDetection as CD_Models
import src.util.Evaluation as Evaluation
import src.util.NER as NER
# import src.util.EntityLinking as EntityLinking
# from genre.trie import Trie, MarisaTrie
import src.util.Results as Results

task = "verifiable-claim-detection"
run_name = "-XLMR"
config = Util.readAllConfig()
path = config["path"]
claim_detection_config = config[task]
model_name = claim_detection_config["default-model"]
max_length = claim_detection_config["max-length"]  # Refer BertTweet Paper

# Read dataset
dataset = ReadDataset.readTrainTestData(path, task, False)

# Get embedding representation
tokenized_input, class_labels, word_embeddings = Embedding.getEmbeddedDataset(path, model_name, dataset, max_length)

# Common training parameters
batch_size = 32
hidden_size = 256
output_size = 2
learning_rate = 3e-5
epochs = 30
MODELS_PATH = path + "Models/Claim-Detection/"
iterations = 10
training_languages = ["english", "dutch", "bulgarian", "turkish"]
we_size = Training.getInputSize(word_embeddings)
ee_size = 256

ner_model = "Babelscape/wikineural-multilingual-ner"
ner_vectors_cg, ner_tagged_sentences_cg = NER.getNERVectors(path, ner_model, dataset, tokenized_input)
entity_indexes, no_entityC = NER.getIndexVector(ner_vectors_cg)

train_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels, training_languages,
                                                    ["train"], batch_size, shuffle=True)
validation_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels,
                                                         training_languages, ["dev"], batch_size,
                                                         shuffle=False)

########################################################################################################################
# Word embedding + Coarse-grained NER
model_prefix = "EClassifier-cNER-" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityC, hidden_size, output_size)
    Training.trainModelWithEntityData(model, train_loader, validation_loader, learning_rate, epochs,
                                      MODELS_PATH + model_name)

    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    Evaluation.evaluateTestDataWithEntity(best_model, model_name, word_embeddings, entity_indexes, dataset,
                                          "test", path)
Results.computeAveragePerformance(path, "test")
logging.info(f"Completed training {model_prefix}")
# ########################################################################################################################
# ee_size = 256
# # Wiki entity extraction using coarse-grained NER
# wiki_entity_scores_cg = EntityLinking.getWikiEntityPresenceScore(path, ner_tagged_sentences_cg, ner_vectors_cg)

# wiki_entity_presence_threshold = -0.15
# entity_indexes, no_entityCW = EntityLinking.getELIndexVector(entity_indexes, wiki_entity_scores_cg,
#                                                              wiki_entity_presence_threshold, no_entityC)


# train_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels, training_languages,
#                                                     ["train"], batch_size, shuffle=True)
# validation_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels,
#                                                          training_languages, ["dev"], batch_size,
#                                                          shuffle=False)

# ########################################################################################################################
# # Word embedding + NER + Wiki Entity Presence
# model_prefix = "EClassifier-NER-cWiki" + str(int(wiki_entity_presence_threshold * 100)) + "-" + run_name
# logging.info(f"Started training {model_prefix}")

# for i in range(iterations):
#     model_name = model_prefix + "-" + str(i) + ".pth"
#     model = CD_Models.EXClaim(we_size, ee_size, no_entityCW, hidden_size, output_size)
#     Training.trainModelWithEntityData(model, train_loader, validation_loader, learning_rate, epochs,
#                                       MODELS_PATH + model_name)

#     best_model = Training.loadModel(model, MODELS_PATH + model_name)
#     Evaluation.evaluateTestDataWithEntity(best_model, model_name, word_embeddings, entity_indexes, dataset,
#                                           "test", path)
# Results.computeAveragePerformance(path, "test")
# logging.info(f"Completed training {model_prefix}")

########################################################################################################################
# Coarse-grained NER models completed. Clear all the variables

ner_vectors_cg = None
ner_tagged_sentences_cg = None
wiki_entities_cg = None
wiki_entity_scores_cg = None
entity_indexes = None
model = None
best_model = None
Util.clearMemory()

########################################################################################################################
ee_size = 128
ner_model = "multinerd-mbert"
ner_vectors_fg, ner_tagged_sentences_fg = NER.getNERVectors(path, ner_model, dataset, tokenized_input)
entity_indexes, no_entityF = NER.getIndexVector(ner_vectors_fg)

train_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels, training_languages,
                                                    ["train"], batch_size, shuffle=True)
validation_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels,
                                                         training_languages, ["dev"], batch_size,
                                                         shuffle=False)

########################################################################################################################
# Word embedding + Fine-grained NER
model_prefix = "EClassifier-fNER-" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityF, hidden_size, output_size)
    Training.trainModelWithEntityData(model, train_loader, validation_loader, learning_rate, epochs,
                                      MODELS_PATH + model_name)

    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    Evaluation.evaluateTestDataWithEntity(best_model, model_name, word_embeddings, entity_indexes, dataset,
                                          "test", path)
Results.computeAveragePerformance(path, "test")
logging.info(f"Completed training {model_prefix}")

# ########################################################################################################################

# # Clear all the variables
# model = None
# best_model = None
# train_loader = None
# validation_loader = None
# Util.clearMemory()

# ########################################################################################################################
# ee_size = 256
# # Wiki entity extraction using fine-grained NER
# wiki_entity_scores_fg = EntityLinking.getWikiEntityPresenceScore(path, ner_tagged_sentences_fg, ner_vectors_fg)

# wiki_entity_presence_threshold = -0.15
# entity_indexes, no_entityFW = EntityLinking.getELIndexVector(entity_indexes, wiki_entity_scores_fg,
#                                                              wiki_entity_presence_threshold, no_entityF)

# train_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels, training_languages,
#                                                     ["train"], batch_size, shuffle=True)
# validation_loader = Training.getDataLoaderWithEntityData(word_embeddings, entity_indexes, class_labels,
#                                                          training_languages, ["dev"], batch_size,
#                                                          shuffle=False)

# ########################################################################################################################
# # Word embedding + NER + Wiki Entity Presence
# model_prefix = "EClassifier-NER-fWiki" + str(int(wiki_entity_presence_threshold * 100)) + "-" + run_name
# logging.info(f"Started training {model_prefix}")

# for i in range(iterations):
#     model_name = model_prefix + "-" + str(i) + ".pth"
#     model = CD_Models.EXClaim(we_size, ee_size, no_entityFW, hidden_size, output_size)
#     Training.trainModelWithEntityData(model, train_loader, validation_loader, learning_rate, epochs,
#                                       MODELS_PATH + model_name)

#     best_model = Training.loadModel(model, MODELS_PATH + model_name)
#     Evaluation.evaluateTestDataWithEntity(best_model, model_name, word_embeddings, entity_indexes, dataset,
#                                           "test", path)
# Results.computeAveragePerformance(path, "test")
# logging.info(f"Completed training {model_prefix}")
