from datasets import load_dataset, load_metric
from pprint import pprint

import treetaggerwrapper as ttwp
import warnings

warnings.filterwarnings("ignore")

tt_fr = ttwp.TreeTagger(TAGLANG="fr")
metric = load_metric("./metrics/seqeval_exhaustive")

perceo_datasets = load_dataset("./datasets/perceo")
label_list = perceo_datasets["train"].features["pos_tags"].feature.names


def evaluate(dataset):
    predicted_tags = []
    actual_tags = []

    for line in dataset:
        tokens = line["tokens"]
        pos_tags = line["pos_tags"]
        pos_predicted = ttwp.make_tags(tt_fr.tag_text(tokens, tagonly=True))

        for i in range(len(tokens)):
            if label_list[pos_tags[i]].startswith(tokens[i]):
                pos_tags[i] = tokens[i]
            if label_list[pos_tags[i]].startswith("AUX"):
                tense = label_list[pos_tags[i]].split(":")[1]
                pos_tags[i] = label_list.index(f"VER:{tense}")

        predicted_tags.append([token.pos if isinstance(token, ttwp.Tag) else "UNK" for token in pos_predicted])
        actual_tags.append([label_list[label_id] for label_id in pos_tags])

    results = metric.compute(predictions=predicted_tags, references=actual_tags)
    return({
        "precision_micro": results["overall_precision_micro"],
        "recall_micro": results["overall_recall_micro"],
        "f1_micro": results["overall_f1_micro"],
        "f1_macro": results["overall_f1_macro"],
        "accuracy": results["overall_accuracy"],
    })


print("With validation set:")
pprint(evaluate(perceo_datasets["validation"]))

print("With test set:")
pprint(evaluate(perceo_datasets["test"]))
