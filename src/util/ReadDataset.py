import os
import pandas as pd
import logging
import src.util.Preprocess as Preprocess

DATA_DIR = "Data/"
logger = logging.getLogger("Cross-lingual-claim-detection")


def readKazemi2021(path, task):
    dataset = {}
    logger.info(f"Reading Kazemi 2021 {task} Data")
    dataset_dir = path + DATA_DIR + "Kazemi-2021/" + task
    languages = ["bn", "en", "hi", "ml", "ta"]

    logger.info(f"Languages existing - {languages}")

    for language in languages:
        test = pd.read_csv(dataset_dir + "/kazemi_" + language + ".csv")
        test.rename(columns={"text": "tweet_text"}, inplace=True)
        logger.info(f"Data size: test - {test.shape[0]}")
        # NOTE: test_gold is assigned as test
        data = {"test": test}
        printDataStatistics(data)
        dataset[language] = data

    return dataset


def readSyntheticTestData(path, languages):
    dataset = {}
    logger.info(f"Reading synthetic test data")
    dataset_dir = path + DATA_DIR + "CheckThat-2022/synthetic/"

    logger.info(f"Languages existing - {languages}")

    for language in languages:
        test = pd.read_csv(dataset_dir + "/verifiable-claim-detection-" + language + ".tsv", sep="\t")
        logger.info(f"Data size: test - {test.shape[0]}")
        # NOTE: test_gold is assigned as test
        data = {"test": test}
        printDataStatistics(data)
        dataset[language] = data

    return dataset

def readCombinedTestData(path):
    """
    Read the combined test data
    :param path: project path
    :return: test data (DataFrame, can be empty)
    """
    # file_path = path + "Data/All/combined_dev_test_predictions.csv"
    file_path = path + DATA_DIR +  "CheckThat-2022/english/verifiable-claim-detection/test.csv"

    # TODO: check whether existing
    if not os.path.exists(file_path):
        logger.warning(f"Combined test data not found: {file_path}")
        return pd.DataFrame()   # return empty

    test_data = pd.read_csv(file_path)
    logger.info(f"Combined test data read. Size - {test_data.shape[0]}")
    return test_data



def mergeTrainTestData(dataset, test_data):
    """
    Merge train and test data
    :param dataset: train data (train, dev, and dev-test)
    :param test_data: test data
    :return: merged dataset
    """
    grouped = test_data.groupby('language')
    for key, value in grouped:
        if key in dataset.keys():
            dataset[key]['test'] = value
        else:
            dataset[key] = {'test': value}
    logger.info("Train and test data merged")
    return dataset


def readTrainTestData(path, task, read_test):
    """
    Read train and test data, combine them and preprocess
    :param path: project path
    :param task: task name
    :param read_test: whether to read test or not
    :return: dataset
    """
    train_data = readCheckThat2022(path, task)

    if read_test:
        test_data = readCombinedTestData(path)
        dataset = mergeTrainTestData(train_data, test_data)
    else:
        dataset = train_data
    dataset = Preprocess.preprocessTweets(dataset)
    return dataset


def readCheckThat2022(path, task):
    """
    Read CheckThat 2022 Dataset task files
    :param path: project directory path
    :param task: task
    :return: dataset in dictionary format e.g. {language: {train, dev, dec_test, test} }
    """
    dataset = {}
    logger.info(f"Reading CheckThat 2022 {task} Task Data")
    dataset_dir = path + DATA_DIR + "CheckThat-2022/"
    languages = [f.name for f in os.scandir(dataset_dir) if f.is_dir()]

    logger.info(f"Languages existing - {languages}")

    for language in languages:
        task_dir = dataset_dir + language + "/" + task
        if os.path.exists(task_dir):
            logger.info(f"Reading {language} data")
            train = pd.read_csv(task_dir + "/train.tsv", sep="\t")
            dev = pd.read_csv(task_dir + "/dev.tsv", sep="\t")
            dev_test = pd.read_csv(task_dir + "/dev_test.tsv", sep="\t")
            dev_test["language"] = language
            dev_test["source"] = "check-that"
            # test = pd.read_csv(task_dir + "/test.tsv", sep="\t")
            # All the test files are combined and handled as a separate dataset for effectiveness
            # test_gold = pd.read_csv(task_dir + "/test_gold.tsv", sep="\t")
            logger.info(f"Data size: train - {train.shape[0]}, dev - {dev.shape[0]}, dev_test - {dev_test.shape[0]}")
            # NOTE: test_gold is assigned as test
            
            data = {"train": train, "dev": dev, "dev_test": dev_test, "test": dev_test}
            printDataStatistics(data)
            dataset[language] = data

    return dataset


def printDataStatistics(data):
    """
    Print class_label statistics of a dataset
    :param data: data in dictionary format e.g. {train, dev, dec_test, test}
    """
    logger.info("Class label distribution: ")
    for key in data:
        distribution = data[key]["class_label"].value_counts()
        logger.info(f"{key} : {distribution}")
