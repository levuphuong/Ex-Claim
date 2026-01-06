import logging
import src.util.Util as Util
import src.util.Embedding as Embedding
import src.util.Training as Training
import src.models.ClaimDetection as CD_Models
import src.util.NER as NER
import src.util.EntityLinking as EntityLinking
from genre.trie import Trie, MarisaTrie
import pandas as pd
import torch
import src.util.ReadDataset as ReadDataset


task = "verifiable-claim-detection"
config = Util.readAllConfig()
path = config["path"]
claim_detection_config = config[task]
model_name = claim_detection_config["default-model"]
max_length = claim_detection_config["max-length"]  # Refer BertTweet Paper

dataset_complete = ReadDataset.readTrainTestData(path, task, True)

# Filter only the test data
dataset = {}
for language in dataset_complete:
    dataset[language] = {"test": dataset_complete[language]["test"]}

# Get embedding representation
tokenized_input, class_labels, word_embeddings = Embedding.getEmbeddedDataset(path, model_name, dataset, max_length)

# Common training parameters
hidden_size = 256
output_size = 2
MODELS_PATH = path + "Models/Claim-Detection/"
RESULTS_PATH = path + "Results/Claim-Detection/"
iterations = 2
we_size = Training.getInputSize(word_embeddings)
ee_size = 128
run_name = "-XLMR"

results = []


def getAttentionWeightsWithEntity(model, input_vectors, entity_indexes):
    device = Util.getDevice()
    model.to(device)
    model.eval()

    X = input_vectors["english"]["test"]
    E = entity_indexes["english"]["test"]

    we = X.to(device)
    e = E.to(device)
    with torch.no_grad():
        ee = model.entity_embedding(e)
        w_projected = model.project1(we)
        e_projected = model.project2(ee)
        x = w_projected + e_projected
        context_vector, weights = model.attention(x, x, x)

    logging.info(f"Attention weight size: {weights.shape}")
    return weights.cpu()

def getAttentionWeights(model, input_vectors):
    device = Util.getDevice()
    model.to(device)
    model.eval()

    X = input_vectors["english"]["test"]

    we = X.to(device)
    with torch.no_grad():
        x = model.projection(we)
        context_vector, weights = model.attention(x, x, x)

    logging.info(f"Attention weight size: {weights.shape}")
    return weights.cpu()


def computeEntropy(weights, mName):
    global results
    entropy = -torch.sum(weights*torch.log(weights + 1e-9), dim=1)
    logging.info(f"Entropy shape {entropy.shape}")
    logging.info(f"Averaging over number of sentences {entropy.shape[0]}")
    average_entropy = torch.sum(entropy)/(128*entropy.shape[0])

    results.append({"Model": mName, "Entropy": average_entropy})

def saveResults():
    global results
    df = pd.DataFrame(results)
    df_average = df.groupby(['Model'])["Entropy"].mean()
    df_average.to_csv(RESULTS_PATH + "entropy.csv")

########################################################################################################################
# Word embedding
model_prefix = "PClassifier" + run_name
logging.info(f"Started training {model_prefix}")
for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.XClaim(we_size, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    attention_weights = getAttentionWeights(best_model, word_embeddings)
    computeEntropy(attention_weights, model_prefix)
logging.info(f"Completed evaluating {model_name}")

########################################################################################################################
ner_model = "Babelscape/wikineural-multilingual-ner"
ner_vectors_cg, ner_tagged_sentences_cg = NER.getNERVectors(path, ner_model, dataset, tokenized_input)
entity_indexes, no_entityC = NER.getIndexVector(ner_vectors_cg)

logging.info(f"Entity Index: {entity_indexes['english']['test'][:,:32]}")
########################################################################################################################
# Word embedding + Coarse-grained NER
model_prefix = "EClassifier-cNER-" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityC, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    attention_weights = getAttentionWeightsWithEntity(best_model, word_embeddings, entity_indexes)
    computeEntropy(attention_weights, model_prefix)
logging.info(f"Completed evaluating {model_name}")

########################################################################################################################

# Clear all the variables
model = None
best_model = None
Util.clearMemory()

########################################################################################################################

# Wiki entity extraction using coarse-grained NER
wiki_entity_scores_cg = EntityLinking.getWikiEntityPresenceScore(path, ner_tagged_sentences_cg, ner_vectors_cg)

wiki_entity_presence_threshold = -0.15

merged = Embedding.concatenateEmbeddings(word_embeddings, ner_vectors_cg)
entity_indexes, no_entityCW = EntityLinking.getELIndexVector(entity_indexes, wiki_entity_scores_cg,
                                                             wiki_entity_presence_threshold, no_entityC)
logging.info(f"Entity Index: {entity_indexes['english']['test'][:,:32]}")
########################################################################################################################
ee_size = 256
# Word embedding + NER + Wiki Entity Presence
model_prefix = "EClassifier-NER-cWiki" + str(int(wiki_entity_presence_threshold * 100)) + "-E256" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityCW, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    attention_weights = getAttentionWeightsWithEntity(best_model, word_embeddings, entity_indexes)
    computeEntropy(attention_weights, model_prefix)
logging.info(f"Completed evaluating {model_name}")

########################################################################################################################
# Coarse-grained NER models completed. Clear all the variables

# Clear all the variables
model = None
best_model = None
ner_vectors_cg = None
ner_tagged_sentences_cg = None
wiki_entities_cg = None
wiki_entity_scores_cg = None
Util.clearMemory()

########################################################################################################################

ner_model = "multinerd-mbert"
ner_vectors_fg, ner_tagged_sentences_fg = NER.getNERVectors(path, ner_model, dataset, tokenized_input)
entity_indexes, no_entityF = NER.getIndexVector(ner_vectors_fg)
logging.info(f"Entity Index: {entity_indexes['english']['test'][:,:32]}")
########################################################################################################################
ee_size = 128
# Word embedding + Fine-grained NER
model_prefix = "EClassifier-fNER-" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityF, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    attention_weights = getAttentionWeightsWithEntity(best_model, word_embeddings, entity_indexes)
    computeEntropy(attention_weights, model_prefix)
logging.info(f"Completed evaluating {model_name}")

########################################################################################################################

# Wiki entity extraction using fine-grained NER
wiki_entity_scores_fg = EntityLinking.getWikiEntityPresenceScore(path, ner_tagged_sentences_fg, ner_vectors_fg)

wiki_entity_presence_threshold = -0.15
entity_indexes, no_entityFW = EntityLinking.getELIndexVector(entity_indexes, wiki_entity_scores_fg,
                                                             wiki_entity_presence_threshold, no_entityF)
logging.info(f"Entity Index: {entity_indexes['english']['test'][:,:32]}")
########################################################################################################################
ee_size = 256
# Word embedding + NER + Wiki Entity Presence
model_prefix = "EClassifier-NER-fWiki" + str(int(wiki_entity_presence_threshold * 100)) + "-" + run_name
logging.info(f"Started training {model_prefix}")

for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.EXClaim(we_size, ee_size, no_entityFW, hidden_size, output_size)
    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    attention_weights = getAttentionWeightsWithEntity(best_model, word_embeddings, entity_indexes)
    computeEntropy(attention_weights, model_prefix)
logging.info(f"Completed evaluating {model_name}")

saveResults()