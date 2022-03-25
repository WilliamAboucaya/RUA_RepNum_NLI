import nltk
import numpy as np
import pandas as pd

from datasets import load_metric
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline

from utils import predict_nli, remove_past_sentences

labeled_proposals_couples = pd.read_csv("../consultation_data/nli_labeled_proposals.csv", encoding="utf8",
                                        engine='python', quoting=3, sep=';', dtype={"label": int})

sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
pos_model = AutoModelForTokenClassification.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
pos_tokenizer = AutoTokenizer.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)

nli_model = AutoModelForSequenceClassification.from_pretrained("waboucay/camembert-base-finetuned-xnli_fr")
nli_tokenizer = AutoTokenizer.from_pretrained("waboucay/camembert-base-finetuned-xnli_fr", model_max_length=512)
accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

labeled_proposals_couples["premise"] = labeled_proposals_couples["premise"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))
labeled_proposals_couples["hypothesis"] = labeled_proposals_couples["hypothesis"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))
labeled_proposals_couples["predicted_label_0.1"] = np.nan
labeled_proposals_couples["predicted_label_0.2"] = np.nan
labeled_proposals_couples["predicted_label_0.3"] = np.nan
labeled_proposals_couples["predicted_label_0.4"] = np.nan
labeled_proposals_couples["predicted_label_0.5"] = np.nan
labeled_proposals_couples["predicted_label_0.6"] = np.nan
labeled_proposals_couples["predicted_label_0.7"] = np.nan
labeled_proposals_couples["predicted_label_0.8"] = np.nan
labeled_proposals_couples["predicted_label_0.9"] = np.nan

with open("../results/contradiction_checking/removepast_sentencewise_contradictionshare.log", "w", encoding="utf8") as file:
    for idx, row in labeled_proposals_couples.iterrows():
        premise_sentences = sentences_tokenizer.tokenize(row["premise"])
        hypothesis_sentences = sentences_tokenizer.tokenize(row["premise"])

        nb_contradictory_pairs = 0

        for i in range(len(premise_sentences)):
            for j in range(len(hypothesis_sentences)):
                predicted_label = predict_nli(row["premise"], row["hypothesis"], nli_tokenizer, nli_model)
                nb_contradictory_pairs += predicted_label

        share_contradictory_pairs = nb_contradictory_pairs / (len(premise_sentences) * len(hypothesis_sentences))

        for threshold in np.arange(0.1, 1, 0.1):
            labeled_proposals_couples.at[idx, f"predicted_label_{threshold}"] = int(share_contradictory_pairs >= threshold)

        if idx % 5 == 0:
            file.write(f'{row["premise"]}\n\n')
        file.write(f'Label: {row["label"]};Nb contradictory pairs: {nb_contradictory_pairs};Share contradictory pairs: {share_contradictory_pairs};{row["hypothesis"]}\n')

        if idx % 5 == 4:
            file.write("===========================================\n\n")

labels = labeled_proposals_couples["label"].tolist()
for threshold in np.arange(0.1, 1, 0.1):
    with open(f"../results/contradiction_checking/removepast_sentencewise_contradictionshare_metrics_{threshold}.log", "w", encoding="utf8") as file:
        predictions = labeled_proposals_couples[f"predicted_label_{threshold}"].tolist()
        file.write("Accuracy: ")
        file.write(str(accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]))
        file.write("\nF1 micro: ")
        file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]))
        file.write("\nF1 macro: ")
        file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]))
