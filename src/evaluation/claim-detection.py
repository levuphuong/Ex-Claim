import logging
import src.util.ReadDataset as ReadDataset
import src.util.Util as Util
import src.util.Embedding as Embedding
import src.util.Training as Training
import src.models.ClaimDetection as CD_Models
import src.util.Evaluation as Evaluation
import src.util.Results as Results

task = "verifiable-claim-detection"
run_name = "XLMR"
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

########################################################################################################################
# Get a single vector representation for the whole sentence

input_vectors = Embedding.getCLSVector(word_embeddings)
input_size = Training.getInputSize(input_vectors)

train_loader = Training.getDataLoader(input_vectors, class_labels, training_languages, ["train"], batch_size,
                                      shuffle=True)
validation_loader = Training.getDataLoader(input_vectors, class_labels, training_languages, ["dev"], batch_size,
                                           shuffle=False)

########################################################################################################################
# FNN, Word embedding
model_prefix = "FNN-" + run_name
logging.info(f"Started training {model_prefix}")
bidirectional = False
for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.FNN(input_size, hidden_size, output_size)
    Training.trainModel(model, train_loader, validation_loader, learning_rate, epochs, MODELS_PATH + model_name)

    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    Evaluation.evaluateTestData(best_model, model_name, input_vectors, dataset, "test", path)
Results.computeAveragePerformance(path, "test")
logging.info(f"Completed training {model_prefix}")

########################################################################################################################

input_vectors = word_embeddings
input_size = Training.getInputSize(input_vectors)

train_loader = Training.getDataLoader(input_vectors, class_labels, training_languages, ["train"], batch_size,
                                      shuffle=True)
validation_loader = Training.getDataLoader(input_vectors, class_labels, training_languages, ["dev"], batch_size,
                                           shuffle=False)

########################################################################################################################

#Word embedding
model_prefix = "PClassifier-" + run_name
logging.info(f"Started training {model_prefix}")
for i in range(iterations):
    model_name = model_prefix + "-" + str(i) + ".pth"
    model = CD_Models.XClaim(input_size, hidden_size, output_size)
    Training.trainModel(model, train_loader, validation_loader, learning_rate, epochs, MODELS_PATH + model_name)

    best_model = Training.loadModel(model, MODELS_PATH + model_name)
    Evaluation.evaluateTestData(best_model, model_name, input_vectors, dataset, "test", path)
Results.computeAveragePerformance(path, "dev_test")
logging.info(f"Completed training {model_prefix}")
