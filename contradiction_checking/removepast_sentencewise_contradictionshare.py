import os
import sys

import nltk
import pandas as pd

from datasets import load_metric
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline

from utils.functions import remove_past_sentences, maximize_f1_score, apply_model_sentencewise


def apply_strategy(proposals_couples: pd.DataFrame, model_checkpoint: str, model_revision: str = "main") -> pd.DataFrame:
    labeled_proposals_couples = proposals_couples.copy()

    sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
    pos_model = AutoModelForTokenClassification.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
    pos_tokenizer = AutoTokenizer.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
    nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)

    try:
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, revision=model_revision)
        nli_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, revision=model_revision, model_max_length=512)
    except OSError:
        print(f"No such revision '{model_revision}' for model '{model_checkpoint}'")
        quit()

    labeled_proposals_couples["premise"] = labeled_proposals_couples["premise"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))
    labeled_proposals_couples["hypothesis"] = labeled_proposals_couples["hypothesis"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))

    labeled_proposals_couples["model_results"] = labeled_proposals_couples.apply(lambda row: apply_model_sentencewise(row, sentences_tokenizer, nli_tokenizer, nli_model), axis=1)

    labeled_proposals_couples["share_contradictory_pairs"] = labeled_proposals_couples.apply(lambda row: row["model_results"][1], axis=1)
    labeled_proposals_couples["nb_contradictory_pairs"] = labeled_proposals_couples.apply(lambda row: row["model_results"][0], axis=1)

    return labeled_proposals_couples


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        input_consultation_name = sys.argv[1]
        input_model_checkpoint = sys.argv[2]
        input_model_revision = sys.argv[3]
    else:
        input_consultation_name = "rua_with_titles_section"
        input_model_checkpoint = "waboucay/camembert-base-finetuned-nli-repnum_wl-rua_wl"
        input_model_revision = "main"

    input_model_name = input_model_checkpoint.split("/")[-1]
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    labeled_proposals = pd.read_csv(f"../consultation_data/nli_labeled_proposals_{input_consultation_name}.csv",
                                    encoding="utf8", engine='python', quoting=0, sep=';', dtype={"label": int})

    if "3_classes" not in input_model_name:
        labeled_proposals["label"] = labeled_proposals["label"].apply(lambda label: 0 if label == 2 else label)

    labeled_proposals = apply_strategy(labeled_proposals, input_model_checkpoint, input_model_revision)

    if not os.path.exists(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}"):
        os.mkdir(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}")

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/removepast_sentencewise_contradictionshare.log",
              "w", encoding="utf8") as file:
        for idx, row in labeled_proposals.iterrows():
            if idx % 5 == 0:
                file.write(f'{row["premise"]}\n\n')
            file.write(f'Label: {row["label"]};Nb contradictory pairs: {row["nb_contradictory_pairs"]};Share contradictory pairs: {row["share_contradictory_pairs"]};{row["hypothesis"]}\n')

            if idx % 5 == 4:
                file.write("===========================================\n\n")

    with open(f"../results/contradiction_checking/{input_consultation_name}/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/removepast_sentencewise_contradictionshare_metrics.log",
              "w", encoding="utf8") as file:
        threshold, max_f1 = maximize_f1_score(labeled_proposals["share_contradictory_pairs"],
                                              labeled_proposals["label"])
        predictions = (labeled_proposals["share_contradictory_pairs"] >= threshold).astype(int).tolist()
        labels = labeled_proposals["label"].tolist()

        file.write(f"With threshold = {threshold}\n")
        file.write("Accuracy: ")
        file.write(str(accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]))
        file.write("\nF1 micro: ")
        file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]))
        file.write("\nF1 macro: ")
        file.write(str(max_f1))
        file.write("\n")
