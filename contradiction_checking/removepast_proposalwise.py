import os
import sys

import nltk
import numpy as np
import pandas as pd

from datasets import load_metric
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline

import sys
sys.path.append('../')

from utils.functions import remove_past_sentences, predict_nli



def apply_strategy(proposals_couples: pd.DataFrame, model_checkpoint: str, model_revision: str = "main") -> pd.DataFrame:
    model_name = model_checkpoint.split("/")[-1]

    labeled_proposals_couples = proposals_couples.copy()

    # sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
    # pos_model = AutoModelForTokenClassification.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
    # pos_tokenizer = AutoTokenizer.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
    # nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)

    try:
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, revision=model_revision)
        nli_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, revision=model_revision, model_max_length=512)
    except OSError:
        print(f"No such revision '{model_revision}' for model '{model_name}'")
        quit()

    labeled_proposals_couples["predicted_label"] = np.nan

    for idx, row in labeled_proposals_couples.iterrows():
        predicted_label = predict_nli(row["premise"], row["hypothesis"], nli_tokenizer, nli_model)
        labeled_proposals_couples.at[idx, "predicted_label"] = predicted_label

    return labeled_proposals_couples


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        input_consultation_name = sys.argv[1]
        input_model_checkpoint = sys.argv[2]
        input_model_revision = sys.argv[3]
    else:
        input_consultation_name = "repnum_with_titles"
        input_model_checkpoint = "waboucay/camembert-large-finetuned-repnum_wl-rua_wl_3_classes"
        input_model_revision = "main"

    input_model_name = input_model_checkpoint.split("/")[-1]
    exp_id = input_model_checkpoint[9:]
    precision_metric = load_metric("precision", experiment_id=exp_id)
    recall_metric = load_metric("recall", experiment_id=exp_id)
    f1_metric = load_metric("f1", experiment_id=exp_id)

    labeled_proposals = pd.read_csv(f"../consultation_data/nli_labeled_proposals_{input_consultation_name}_nopast.csv",
                                    encoding="utf8", engine='python', quoting=0, sep=';', dtype={"label": int})

    labeled_proposals = apply_strategy(labeled_proposals, input_model_checkpoint, input_model_revision)

    if not os.path.exists(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}"):
        os.mkdir(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}")

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/removepast_proposalwise.log", "w", encoding="utf8") as file:
        for idx, row in labeled_proposals.iterrows():
            if idx % 5 == 0:
                file.write(f'{row["premise"]}\n\n')
            file.write(f'Label: {row["label"]};Prediction: {row["predicted_label"]};{row["hypothesis"]}\n')

            if idx % 5 == 4:
                file.write("===========================================\n\n")

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/removepast_proposalwise_metrics.log", "w", encoding="utf8") as file:
        predictions = labeled_proposals["predicted_label"].tolist()
        labels = labeled_proposals["label"].tolist()

        ConfusionMatrixDisplay.from_predictions(labels, predictions)
        plt.savefig(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/removepast_proposalwise_matrix.eps", format="eps")
        plt.show()

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
