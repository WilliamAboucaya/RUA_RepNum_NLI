import os
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

from datasets import load_metric
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline
import sys
sys.path.append('../')

import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from utils.functions import remove_past_sentences, apply_model_sentencecouple_batch, define_label


def apply_strategy(proposals_couples: pd.DataFrame, model_checkpoint: str, model_revision: str, batch_size) -> pd.DataFrame:
    model_name = model_checkpoint.split("/")[-1]

    labeled_proposals_couples = proposals_couples.copy()

    sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
    pos_model = AutoModelForTokenClassification.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
    pos_tokenizer = AutoTokenizer.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
    nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)

    try:
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, revision=model_revision).to(device)
        nli_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, revision=model_revision, model_max_length=512)
    except OSError:
        print(f"No such revision '{model_revision}' for model '{model_name}'")
        quit()

    labeled_proposals_couples["premise"] = labeled_proposals_couples["premise"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))
    labeled_proposals_couples["hypothesis"] = labeled_proposals_couples["hypothesis"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))

    apply_model_sentencecouple_batch(labeled_proposals_couples, sentences_tokenizer, nli_tokenizer, nli_model, batch_size)

    return labeled_proposals_couples


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        input_consultation_name = sys.argv[1]
        input_model_checkpoint = sys.argv[2]
        input_model_revision = sys.argv[3]
        batch_size = int(sys.argv[4])
    else:
        input_consultation_name = "repnum_with_titles"
        input_model_checkpoint = "waboucay/camembert-large-finetuned-xnli_fr_3_classes-finetuned-rua_wl_3_classes"
        input_model_revision = "main"
        batch_size = 8

    input_model_name = input_model_checkpoint.split("/")[-1]

    exp_id = input_model_checkpoint[9:]
    precision_metric = load_metric("precision", experiment_id=exp_id)
    recall_metric = load_metric("recall", experiment_id=exp_id)
    f1_metric = load_metric("f1", experiment_id=exp_id)

    labeled_proposals = pd.read_csv(f"../consultation_data/nli_labeled_proposals_{input_consultation_name}_nopast.csv",
                                    encoding="utf8", engine='python', quoting=0, sep=';', dtype={"label": int})

    labeled_proposals = apply_strategy(labeled_proposals, input_model_checkpoint, input_model_revision, batch_size)

    consultation_prefix = input_consultation_name.split("_")[0]

    if not os.path.exists(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}"):
        os.mkdir(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}")

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/removepast_sentencecouple_contradictionshare.log",
              "w", encoding="utf8") as file:
        for idx, row in labeled_proposals.iterrows():
            if idx % 5 == 0:
                file.write(f'{row["premise"]}\n\n')
            file.write(f'Label: {row["label"]};Nb contradictory pairs: {row["nb_contradictory_pairs"]};Share contradictory pairs: {row["share_contradictory_pairs"]};Nb entailed pairs: {row["nb_entailed_pairs"]};Share entailed pairs: {row["share_entailed_pairs"]};Nb neutral pairs: {row["nb_neutral_pairs"]};Share neutral pairs: {row["share_neutral_pairs"]};{row["hypothesis"]}\n')

            if idx % 5 == 4:
                file.write("===========================================\n\n")

    with open(f"../results/threshold/{consultation_prefix}_nli/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/removepast_sentencecouple_contradictionshare.log", "r", encoding="utf8") as file:
        computed_contradiction_threshold = float(re.findall("\d+\.\d+", file.readline())[0])
        computed_entailment_threshold = float(re.findall("\d+\.\d+", file.readline())[0])

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/removepast_sentencecouple_contradictionshare_metrics.log",
              "w", encoding="utf8") as file:
        for contradiction_threshold in np.append(computed_contradiction_threshold, np.arange(0.1, 1, 0.1)):
            predictions = labeled_proposals.apply(
                lambda row: define_label(row["share_contradictory_pairs"], row["share_entailed_pairs"],
                                         contradiction_threshold, computed_entailment_threshold), axis=1).tolist()
            labels = labeled_proposals["label"].tolist()

            if contradiction_threshold == computed_contradiction_threshold:
                ConfusionMatrixDisplay.from_predictions(labels, predictions)
                plt.tight_layout()
                plt.savefig(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/removepast_sentencecouple_contradictionshare_matrix.eps", format="eps")
                plt.show()

            file.write(f"With contradiction_threshold = {contradiction_threshold} and entailment_threshold = {computed_entailment_threshold}{' * COMPUTED THRESHOLDS' if contradiction_threshold == computed_contradiction_threshold else ''}\n")
            precision_results = precision_metric.compute(predictions=predictions, references=labels, average=None)["precision"]
            recall_results = recall_metric.compute(predictions=predictions, references=labels, average=None)["recall"]
            file.write("Precision: ")
            file.write(f"{precision_results[0]} for label 0 | {precision_results[1]} for label 1 | {precision_results[2]} for label 2")
            file.write("\nRecall: ")
            file.write(f"{recall_results[0]} for label 0 | {recall_results[1]} for label 1 | {recall_results[2]} for label 2")
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

            file.write(f"With contradiction_threshold = {computed_contradiction_threshold} and entailment_threshold = {entailment_threshold}\n")
            precision_results = precision_metric.compute(predictions=predictions, references=labels, average=None)["precision"]
            recall_results = recall_metric.compute(predictions=predictions, references=labels, average=None)["recall"]
            file.write("Precision: ")
            file.write(f"{precision_results[0]} for label 0 | {precision_results[1]} for label 1 | {precision_results[2]} for label 2")
            file.write("\nRecall: ")
            file.write(f"{recall_results[0]} for label 0 | {recall_results[1]} for label 1 | {recall_results[2]} for label 2")
            file.write("\nF1 micro: ")
            file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]))
            file.write("\nF1 macro: ")
            file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]))
            file.write("\n")
