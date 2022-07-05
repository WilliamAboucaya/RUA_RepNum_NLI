import re
import sys

import joblib
import nltk
import numpy as np
import pandas as pd
import os

from datasets import load_metric
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
sys.path.append('../')

from utils.functions import maximize_f1_score, apply_model_sentencecouple, define_label


def apply_strategy(proposals_couples: pd.DataFrame, model_checkpoint: str, model_revision: str = "main") -> pd.DataFrame:
    model_name = model_checkpoint.split("/")[-1]

    labeled_proposals_couples = proposals_couples.copy()

    sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")

    try:
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, revision=model_revision)
        nli_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, revision=model_revision, model_max_length=512)
    except OSError:
        print(f"No such revision '{model_revision}' for model '{model_name}'")
        quit()

    labeled_proposals_couples["model_results"] = labeled_proposals_couples.apply(lambda row: apply_model_sentencecouple(row, sentences_tokenizer, nli_tokenizer, nli_model), axis=1)

    labeled_proposals_couples["nb_entailed_pairs"] = labeled_proposals_couples.apply(lambda row: row["model_results"][0], axis=1)
    labeled_proposals_couples["share_entailed_pairs"] = labeled_proposals_couples.apply(lambda row: row["model_results"][1], axis=1)
    labeled_proposals_couples["nb_contradictory_pairs"] = labeled_proposals_couples.apply(lambda row: row["model_results"][2], axis=1)
    labeled_proposals_couples["share_contradictory_pairs"] = labeled_proposals_couples.apply(lambda row: row["model_results"][3], axis=1)
    labeled_proposals_couples["nb_neutral_pairs"] = labeled_proposals_couples.apply(lambda row: row["model_results"][4], axis=1)
    labeled_proposals_couples["share_neutral_pairs"] = labeled_proposals_couples.apply(lambda row: row["model_results"][5], axis=1)

    return labeled_proposals_couples


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        input_consultation_name = sys.argv[1]
        input_model_checkpoint = sys.argv[2]
        input_model_revision = sys.argv[3]
    else:
        input_consultation_name = "repnum_with_titles"
        input_model_checkpoint = "waboucay/camembert-large-finetuned-repnum_wl-rua_wl"
        input_model_revision = "main"

    input_model_name = input_model_checkpoint.split("/")[-1]
    input_consultation_prefix = input_consultation_name.split("_")[0]

    exp_id = input_model_checkpoint[9:]
    precision_metric = load_metric("precision", experiment_id=exp_id)
    recall_metric = load_metric("recall", experiment_id=exp_id)
    f1_metric = load_metric("f1", experiment_id=exp_id)

    labeled_proposals = pd.read_csv(f"../consultation_data/nli_labeled_proposals_{input_consultation_name}.csv",
                                            encoding="utf8", engine='python', quoting=0, sep=';', dtype={"label": int})

    labeled_proposals = apply_strategy(labeled_proposals, input_model_checkpoint, input_model_revision)

    consultation_prefix = input_consultation_name.split("_")[0]

    if not os.path.exists(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}"):
        os.mkdir(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}")

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/withpast_sentencecouple_contradictionshare.log", "w", encoding="utf8") as file:
        for idx, row in labeled_proposals.iterrows():
            if idx % 5 == 0:
                file.write(f'{row["premise"]}\n\n')
            file.write(f'Label: {row["label"]};Nb contradictory pairs: {row["nb_contradictory_pairs"]};Share contradictory pairs: {row["share_contradictory_pairs"]};Nb entailed pairs: {row["nb_entailed_pairs"]};Share entailed pairs: {row["share_entailed_pairs"]};Nb neutral pairs: {row["nb_neutral_pairs"]};Share neutral pairs: {row["share_neutral_pairs"]};{row["hypothesis"]}\n')
            if idx % 5 == 4:
                file.write("===========================================\n\n")

    with open(f"../results/threshold/{consultation_prefix}_nli/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/withpast_sentencecouple_contradictionshare.log", "r", encoding="utf8") as file:
        computed_contradiction_threshold = float(re.findall("\d+\.\d+", file.readline())[0])
        computed_entailment_threshold = float(re.findall("\d+\.\d+", file.readline())[0])

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/withpast_sentencecouple_contradictionshare_metrics.log", "w", encoding="utf8") as file:
        # threshold, max_f1 = maximize_f1_score(labeled_proposals["share_contradictory_pairs"],
        #                                       labeled_proposals["label"])
        for contradiction_threshold in np.append(computed_contradiction_threshold, np.arange(0.1, 1, 0.1)):
            predictions = labeled_proposals.apply(
                lambda row: define_label(row["share_contradictory_pairs"], row["share_entailed_pairs"],
                                         contradiction_threshold, computed_entailment_threshold), axis=1).tolist()
            labels = labeled_proposals["label"].tolist()

            if contradiction_threshold == computed_contradiction_threshold:
                ConfusionMatrixDisplay.from_predictions(np.ndarray(labels), np.ndarray(predictions))

                plt.savefig(f"../results/threshold/{consultation_prefix}_nli/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/removepast_sentencecouple_contradictionshare_matrix.eps", format="eps")
                plt.plot()

            file.write(f"With contradiction_threshold = {contradiction_threshold} and entailment_threshold = {computed_entailment_threshold}{' * COMPUTED THRESHOLDS' if contradiction_threshold == computed_contradiction_threshold else ''}\n")
            file.write("Precision: ")
            file.write(str(precision_metric.compute(predictions=predictions, references=labels)["precision"]))
            file.write("Recall: ")
            file.write(str(recall_metric.compute(predictions=predictions, references=labels)["recall"]))
            file.write("\nF1 micro: ")
            file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]))
            file.write("\nF1 macro: ")
            file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]))
            file.write("\n")
        for entailment_threshold in np.arange(0.1, 1, 0.1):
            predictions = labeled_proposals.apply(
                lambda row: define_label(row["share_contradictory_pairs"], row["share_entailed_pairs"],
                                         computed_contradiction_threshold, entailment_threshold), axis=1).tolist()
            labels = labeled_proposals["label"].tolist()

            file.write(
                f"With contradiction_threshold = {computed_contradiction_threshold} and entailment_threshold = {entailment_threshold}\n")
            file.write("Precision: ")
            file.write(str(precision_metric.compute(predictions=predictions, references=labels)["precision"]))
            file.write("Recall: ")
            file.write(str(recall_metric.compute(predictions=predictions, references=labels)["recall"]))
            file.write("\nF1 micro: ")
            file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]))
            file.write("\nF1 macro: ")
            file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]))
            file.write("\n")
