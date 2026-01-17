import pandas as pd
import numpy as np
import re
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger("Cross-lingual-claim-detection")


def getModelNames(test_data):
    """
    Get list of models from the prediction file
    :param test_data: prediction data
    :return: list of models
    """
    column_names = test_data.columns
    pattern = r'(.+)-\d+$'
    model_columns = [col for col in column_names if re.match(pattern, col)]
    models = list(set([re.match(pattern, col).group(1) for col in model_columns]))

    logging.info(f"Models analysed: {models}")
    return models

iterations = 1

def getLanguageLevelResults(test_data, models, columns_to_mean):
    """
    Get language level average performance of the models
    :param test_data: data with predictions
    :param models: list of model names
    :param columns_to_mean: columns to compute the average
    :return: dataframe with language level average results
    """
    languages = test_data['language'].unique()
    language_results = []

    for language in languages:
        language_data = test_data[test_data['language'] == language]
        source = language_data.iloc[0]['source']
        true_prediction = language_data['class_label']
        for model in models:
            accuracyL = []
            precisionL = []
            recallL = []
            f1_scoresL = []

            for i in range(iterations):
                col_name = f"{model}-{i}"
                col_name = col_name.replace("--", "-")
                language_data.columns = language_data.columns.str.replace("--", "-")

                logger.info(f"Columns available: {language_data.columns.tolist()}")
                logger.info(f"Need column: {col_name}")

                predictions = language_data[col_name]
                accuracy = accuracy_score(true_prediction, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(true_prediction, predictions,
                                                                           average="weighted", zero_division=0)
                accuracyL.append(accuracy)
                precisionL.append(precision)
                recallL.append(recall)
                f1_scoresL.append(f1)

            language_results.append(
                [source, language, model, np.mean(accuracyL), np.std(accuracyL), np.mean(precisionL),
                 np.std(precisionL), np.mean(recallL), np.std(recallL), np.mean(f1_scoresL), np.std(f1_scoresL)])

    results = pd.DataFrame(np.array(language_results), columns=["source", "language", "model", "accuracy",
                                                                "accuracy-std", "precision", "precision-std",
                                                                "recall", "recall-std", "f1_score", "f1_score-std"])
    for col in columns_to_mean:
        results[col] = results[col].astype(float)

    return results


def saveLanguageLevelPerformance(results, path):
    """
    Save language level performance of the models
    :param results: language level performance
    :param path: project path
    """
    # Write to Excel file
    excel_writer = pd.ExcelWriter(path + 'Results/Claim-Detection/language_level_performance_wa.xlsx',
                                  engine='xlsxwriter')
    for language, group in results.groupby('language'):
        group.drop('language', axis=1).to_excel(excel_writer, sheet_name=language, index=False)
    excel_writer.close()

    results_f1_score = results[["language", "model", "f1_score"]]
    results_f1_score = results_f1_score.pivot(index="language", columns="model", values="f1_score")
    results_f1_score = results_f1_score[sorted(results_f1_score.columns, reverse=True)]
    results_f1_score = results_f1_score.reset_index()

    excel_writer = pd.ExcelWriter(path + 'Results/Claim-Detection/language_level_f1_score_wa.xlsx', engine='xlsxwriter')
    results_f1_score.to_excel(excel_writer, index=False)
    excel_writer.close()


def saveSourceLevelResults(results, path, columns_to_mean, data_type):
    """
    Save source level performance of the models
    :param results: language level performance used to compute source level performance
    :param path: project path
    :param columns_to_mean: columns to compute the average
    :param data_type: test or dev_test
    """

    if data_type == "test":
        file_name = "source_level_performance_wa.xlsx"
        results_copy = results.copy()
        results_copy['source'] = 'all'
        results = pd.concat([results, results_copy], ignore_index=True)
    else:
        file_name = "parameter_tuning_wa.xlsx"

    source_wise_results = results.groupby(['source', 'model'])[columns_to_mean].mean()
    source_wise_results = source_wise_results.reset_index()

    # Write to Excel file
    excel_writer = pd.ExcelWriter(path + 'Results/Claim-Detection/' + file_name,
                                  engine='xlsxwriter')
    for source, group in source_wise_results.groupby('source'):
        group.drop('source', axis=1).sort_values(by="model", ascending=False).to_excel(excel_writer, sheet_name=source,
                                                                                       index=False)
    excel_writer.close()


def computeAveragePerformance(path, data_type):
    """
    Generate average performance of the models
    :param path: project path
    :param data_type: test or dev_test
    """
    test_data = pd.read_csv(path + "Data/All/combined_" + data_type + "_predictions.csv")
    models = getModelNames(test_data)

    columns_to_mean = ["accuracy", "accuracy-std", "precision", "precision-std", "recall", "recall-std", "f1_score",
                       "f1_score-std"]

    results = getLanguageLevelResults(test_data, models, columns_to_mean)

    if data_type == "test":
        saveLanguageLevelPerformance(results, path)

    saveSourceLevelResults(results, path, columns_to_mean, data_type)
