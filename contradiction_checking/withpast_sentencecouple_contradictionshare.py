import sys

import joblib
import nltk
import pandas as pd
import os

from datasets import load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.functions import maximize_f1_score, apply_model_sentencecouple


def apply_strategy(proposals_couples: pd.DataFrame, model_checkpoint: str, model_revision: str = "main") -> pd.DataFrame:
    labeled_proposals_couples = proposals_couples.copy()

    sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")

    try:
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, revision=model_revision)
        nli_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, revision=model_revision, model_max_length=512)
    except OSError:
        print(f"No such revision '{model_revision}' for model '{model_checkpoint}'")
        quit()

    labeled_proposals_couples["model_results"] = labeled_proposals_couples.apply(lambda row: apply_model_sentencecouple(row, sentences_tokenizer, nli_tokenizer, nli_model), axis=1)

    labeled_proposals_couples["share_contradictory_pairs"] = labeled_proposals_couples.apply(lambda row: row["model_results"][1], axis=1)
    labeled_proposals_couples["nb_contradictory_pairs"] = labeled_proposals_couples.apply(lambda row: row["model_results"][0], axis=1)

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
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    labeled_proposals = pd.read_csv(f"../consultation_data/nli_labeled_proposals_{input_consultation_name}.csv",
                                            encoding="utf8", engine='python', quoting=0, sep=';', dtype={"label": int})

    if "3_classes" not in input_model_name:
        labeled_proposals["label"] = labeled_proposals["label"].apply(lambda label: 0 if label == 2 else label)

    labeled_proposals = apply_strategy(labeled_proposals, input_model_checkpoint, input_model_revision)

    clf = joblib.load(f"../results/joblib_dumps/{input_consultation_prefix}_nli/classifier_{input_model_name}_withpast_sentencecouple_contradictionshare.joblib")
    labeled_proposals_predictions = clf.predict(labeled_proposals[["share_contradictory_pairs", "nb_contradictory_pairs"]].to_numpy())

    if not os.path.exists(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}"):
        os.mkdir(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}")

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/withpast_sentencecouple_contradictionshare.log", "w", encoding="utf8") as file:
        for idx, row in labeled_proposals.iterrows():
            if idx % 5 == 0:
                file.write(f'{row["premise"]}\n\n')
            file.write(f'Label: {row["label"]};Nb contradictory pairs: {row["nb_contradictory_pairs"]};Share contradictory pairs: {row["share_contradictory_pairs"]};{row["hypothesis"]}\n')

            if idx % 5 == 4:
                file.write("===========================================\n\n")

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/withpast_sentencecouple_contradictionshare_metrics.log", "w", encoding="utf8") as file:
        # threshold, max_f1 = maximize_f1_score(labeled_proposals["share_contradictory_pairs"],
        #                                       labeled_proposals["label"])
        # predictions = (labeled_proposals["share_contradictory_pairs"] >= threshold).astype(int).tolist()
        predictions = labeled_proposals_predictions.tolist()
        labels = labeled_proposals["label"].tolist()

        # file.write(f"With threshold = {threshold}\n")
        file.write("Accuracy: ")
        file.write(str(accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]))
        file.write("\nF1 micro: ")
        file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]))
        file.write("\nF1 macro: ")
        file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]))
        file.write("\n")