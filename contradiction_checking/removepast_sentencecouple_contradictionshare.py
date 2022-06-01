import os
import sys

import nltk
import numpy as np
import pandas as pd

from datasets import load_metric
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline

from utils.functions import predict_nli, remove_past_sentences

if len(sys.argv) >= 4:
    consultation_name = sys.argv[1]
    model_checkpoint = sys.argv[2]
    model_revision = sys.argv[3]
else:
    consultation_name = "rua_with_titles_section"
    model_checkpoint = "waboucay/camembert-base-finetuned-nli-repnum_wl-rua_wl"
    model_revision = "main"

model_name = model_checkpoint.split("/")[-1]

labeled_proposals_couples = pd.read_csv(f"../consultation_data/nli_labeled_proposals_{consultation_name}.csv", encoding="utf8",
                                        engine='python', quoting=0, sep=';', dtype={"label": int})

sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
pos_model = AutoModelForTokenClassification.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
pos_tokenizer = AutoTokenizer.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)

nli_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, revision=model_revision)
nli_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, revision=model_revision, model_max_length=512)
accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

if not os.path.exists(f"../results/contradiction_checking/{consultation_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}"):
    os.mkdir(f"../results/contradiction_checking/{consultation_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}")

labeled_proposals_couples["premise"] = labeled_proposals_couples["premise"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))
labeled_proposals_couples["hypothesis"] = labeled_proposals_couples["hypothesis"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))
for threshold in np.arange(0.1, 1, 0.1):
    labeled_proposals_couples[f"predicted_label_{threshold}"] = np.nan

with open(f"../results/contradiction_checking/{consultation_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}/removepast_sentencecouple_contradictionshare.log", "w", encoding="utf8") as file:
    for idx, row in labeled_proposals_couples.iterrows():
        premise_sentences = sentences_tokenizer.tokenize(row["premise"])
        hypothesis_sentences = sentences_tokenizer.tokenize(row["hypothesis"])

        nb_contradictory_pairs = 0

        for i in range(1, len(premise_sentences)):
            for j in range(1, len(hypothesis_sentences)):
                predicted_label = predict_nli(" ".join(premise_sentences[i - 1:i + 1]),
                                              " ".join(hypothesis_sentences[j - 1:j + 1]), nli_tokenizer, nli_model)
                nb_contradictory_pairs += predicted_label

        try:
            share_contradictory_pairs = nb_contradictory_pairs / ((len(premise_sentences) - 1) * (len(hypothesis_sentences) - 1))
        except ZeroDivisionError:
            share_contradictory_pairs = 0

        for threshold in np.arange(0.1, 1, 0.1):
            labeled_proposals_couples.at[idx, f"predicted_label_{threshold}"] = int(share_contradictory_pairs >= threshold)

        if idx % 5 == 0:
            file.write(f'{row["premise"]}\n\n')
        file.write(f'Label: {row["label"]};Nb contradictory pairs: {nb_contradictory_pairs};Share contradictory pairs: {share_contradictory_pairs};{row["hypothesis"]}\n')

        if idx % 5 == 4:
            file.write("===========================================\n\n")

with open(f"../results/contradiction_checking/{consultation_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}/{model_name}/removepast_sentencecouple_contradictionshare_metrics.log", "w", encoding="utf8") as file:
    labels = labeled_proposals_couples["label"].tolist()
    for threshold in np.arange(0.1, 1, 0.1):
        predictions = labeled_proposals_couples[f"predicted_label_{threshold}"].tolist()
        file.write(f"With threshold = {threshold}\n")
        file.write("Accuracy: ")
        file.write(str(accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]))
        file.write("\nF1 micro: ")
        file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]))
        file.write("\nF1 macro: ")
        file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]))
        file.write("\n")
