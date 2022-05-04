import nltk
import numpy as np
import pandas as pd

from datasets import load_metric
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline

from utils import predict_nli

consultation_name = "repnum"
model_checkpoint = "waboucay/camembert-base-finetuned-nli-repnum_wl-rua_wl"
model_name = model_checkpoint.split("/")[-1]

labeled_proposals_couples = pd.read_csv(f"../consultation_data/nli_labeled_proposals_{consultation_name}.csv", encoding="utf8",
                                        engine='python', quoting=0, sep=';', dtype={"label": int})

sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
pos_model = AutoModelForTokenClassification.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
pos_tokenizer = AutoTokenizer.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)

nli_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
nli_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)
accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

labeled_proposals_couples["predicted_label"] = np.nan

with open(f"../results/contradiction_checking/{consultation_name}/{model_name}/withpast_proposalwise.log", "w", encoding="utf8") as file:
    for idx, row in labeled_proposals_couples.iterrows():
        predicted_label = predict_nli(row["premise"], row["hypothesis"], nli_tokenizer, nli_model)
        labeled_proposals_couples.at[idx, "predicted_label"] = predicted_label
        if idx % 5 == 0:
            file.write(f'{row["premise"]}\n\n')
        file.write(f'Label: {row["label"]};Prediction: {predicted_label};{row["hypothesis"]}\n')

        if idx % 5 == 4:
            file.write("===========================================\n\n")

with open(f"../results/contradiction_checking/{consultation_name}/{model_name}/withpast_proposalwise_metrics.log", "w", encoding="utf8") as file:
    predictions = labeled_proposals_couples["predicted_label"].tolist()
    labels = labeled_proposals_couples["label"].tolist()
    file.write("Accuracy: ")
    file.write(str(accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]))
    file.write("\nF1 micro: ")
    file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]))
    file.write("\nF1 macro: ")
    file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]))
