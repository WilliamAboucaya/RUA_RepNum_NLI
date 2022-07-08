import math
import os

import numpy as np
import pandas as pd

from datasets import load_metric
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys
sys.path.append('../')

import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from utils.functions import predict_nli_batch


def apply_strategy(proposals_couples: pd.DataFrame, model_checkpoint: str, model_revision: str, batch_size: int) -> pd.DataFrame:
    model_name = model_checkpoint.split("/")[-1]

    labeled_proposals_couples = proposals_couples.copy()

    try:
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, revision=model_revision).to(device)
        nli_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, revision=model_revision, model_max_length=512)
    except OSError:
        print(f"No such revision '{model_revision}' for model '{model_name}'")
        quit()

    labeled_proposals_couples["predicted_label"] = np.nan

    nb_batches = int(math.ceil(len(labeled_proposals_couples) / batch_size))

    for i in range(nb_batches):
        start_poz = i * batch_size
        stop_poz = min(start_poz + batch_size, len(labeled_proposals_couples))
        batch = []
        for j in range(start_poz, stop_poz):
            batch.append((labeled_proposals_couples.at[j, "premise"], labeled_proposals_couples.at[j, "hypothesis"]))

        predicted_labels = predict_nli_batch(batch, nli_tokenizer, nli_model)
        for j in range(start_poz, stop_poz):
            labeled_proposals_couples.at[j, "predicted_label"] = predicted_labels[j - start_poz].item()

    return labeled_proposals_couples


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        input_consultation_name = sys.argv[1]
        input_model_checkpoint = sys.argv[2]
        input_model_revision = sys.argv[3]
        batch_size = int(sys.argv[4])
    else:
        input_consultation_name = "repnum"
        input_model_checkpoint = "waboucay/camembert-base-finetuned-nli-repnum_wl-rua_wl"
        input_model_revision = "main"
        batch_size = 8

    input_model_name = input_model_checkpoint.split("/")[-1]

    exp_id = input_model_checkpoint[9:]
    precision_metric = load_metric("precision", experiment_id=exp_id)
    recall_metric = load_metric("recall", experiment_id=exp_id)
    f1_metric = load_metric("f1", experiment_id=exp_id)

    labeled_proposals = pd.read_csv(f"../consultation_data/nli_labeled_proposals_{input_consultation_name}.csv", encoding="utf8",
                                                engine='python', quoting=0, sep=';', dtype={"label": int})

    labeled_proposals = apply_strategy(labeled_proposals, input_model_checkpoint, input_model_revision, batch_size)

    if not os.path.exists(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}"):
        os.mkdir(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}")

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/withpast_proposalwise.log", "w", encoding="utf8") as file:
        for idx, row in labeled_proposals.iterrows():
            if idx % 5 == 0:
                file.write(f'{row["premise"]}\n\n')
            file.write(f'Label: {row["label"]};Prediction: {row["predicted_label"]};{row["hypothesis"]}\n')

            if idx % 5 == 4:
                file.write("===========================================\n\n")

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/withpast_proposalwise_metrics.log", "w", encoding="utf8") as file:
        predictions = labeled_proposals["predicted_label"].tolist()
        labels = labeled_proposals["label"].tolist()

        ConfusionMatrixDisplay.from_predictions(labels, predictions)
        plt.tight_layout()
        plt.savefig(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/withpast_proposalwise_matrix.eps", format="eps")
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
