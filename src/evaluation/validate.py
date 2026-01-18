import logging
import src.util.ReadDataset as ReadDataset
import src.util.Util as Util
import src.util.Embedding as Embedding
import src.models.ClaimDetection as CD_Models
import src.util.NER as NER
import src.util.Evaluation as Evaluation
import src.util.Results as Results
import src.util.Training as Training
from src.util.PHoNER import getPhoNERIndexVector, getPhoNERVectors

# ==== Config ====
task = "verifiable-claim-detection"
config = Util.readAllConfig()
path = config["path"]
claim_cfg = config[task]
model_name = claim_cfg["default-model"]
max_length = claim_cfg["max-length"]
MODELS_PATH = path + "Models/Claim-Detection/"

# ==== Dataset ====
dataset = ReadDataset.readTestData(path, task)

# ==== Embeddings ====
tokenized_input, class_labels, word_embeddings = \
    Embedding.getEmbeddedDataset(path, model_name, dataset, max_length)

# ==== NER ====
# ner_model = "Babelscape/wikineural-multilingual-ner"
# ner_model = "NlpHUST/ner-vibert4news"
# ner_vectors, ner_tags = NER.getNERVectors(path, ner_model, dataset, tokenized_input)
# entity_indexes, no_entity = NER.getIndexVector(ner_vectors)

ner_vectors, ner_tags = getPhoNERVectors(path, "phonlp", dataset, tokenized_input)
entity_indexes, no_entity = getPhoNERIndexVector(ner_vectors)


# ==== Load model ====
# model_file = "EClassifier-cNER--XLMR-0.pth"   # chỉnh số iteration nếu cần
model = CD_Models.EXClaim(
    we_size=Training.getInputSize(word_embeddings),
    ee_size=256,
    no_entity=no_entity,
    hidden_size=256,
    output_size=2
)
# model = Training.loadModel(model, MODELS_PATH + model_file)

# ==== Evaluate ====
run_name = "-XLMR"
model_prefix = "EClassifier-cNER-" + run_name
logging.info(f"Started training {model_prefix}")
iterations = 10
for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"

    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    Evaluation.evaluateTestDataWithEntity(
        model, model_name, word_embeddings, entity_indexes, dataset, "test", path
    )

# ==== Metrics ====
Results.computeAveragePerformance(path, "test")
logging.info("Completed validation.")
