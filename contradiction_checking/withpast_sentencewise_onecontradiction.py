import nltk
import numpy as np
import pandas as pd

from datasets import load_metric
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline

from utils import predict_nli

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

labeled_proposals_couples["predicted_label"] = np.nan

with open("../results/contradiction_checking/withpast_sentencewise_onecontradiction.log", "w", encoding="utf8") as file:
    for idx, row in labeled_proposals_couples.iterrows():
        premise_sentences = sentences_tokenizer.tokenize(row["premise"])
        hypothesis_sentences = sentences_tokenizer.tokenize(row["premise"])

        has_contradictory_pairs = False

        for i in range(len(premise_sentences)):
            for j in range(len(hypothesis_sentences)):
                predicted_label = predict_nli(row["premise"], row["hypothesis"], nli_tokenizer, nli_model)
                if predicted_label:
                    has_contradictory_pairs = True
                    break
            if has_contradictory_pairs:
                break

        labeled_proposals_couples.at[idx, "predicted_label"] = int(has_contradictory_pairs)

        if idx % 5 == 0:
            file.write(f'{row["premise"]}\n\n')
        file.write(f'Label: {row["label"]};Has contradictory pairs: {has_contradictory_pairs};{row["hypothesis"]}\n')

        if idx % 5 == 4:
            file.write("===========================================\n\n")

with open("../results/contradiction_checking/withpast_sentencewise_onecontradiction_metrics.log", "w", encoding="utf8") as file:
    predictions = labeled_proposals_couples["predicted_label"].tolist()
    labels = labeled_proposals_couples["label"].tolist()
    file.write("Accuracy: ")
    file.write(str(accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]))
    file.write("\nF1 micro: ")
    file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]))
    file.write("\nF1 macro: ")
    file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]))
