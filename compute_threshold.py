import importlib
import os
import sys
import joblib

from datasets import load_dataset, load_metric
from sklearn.linear_model import LogisticRegression

from utils.functions import maximize_f1_score, define_label

if len(sys.argv) >= 5:
    dataset_name = sys.argv[1]
    model_checkpoint = sys.argv[2]
    model_revision = sys.argv[3]
    strategy_to_apply = sys.argv[4]
else:
    dataset_name = "repnum_nli"
    model_checkpoint = "waboucay/camembert-large-finetuned-repnum_wl-rua_wl"
    model_revision = "main"
    strategy_to_apply = "removepast_sentencecouple_contradictionshare"

model_name = model_checkpoint.split("/")[-1]
dataset = load_dataset(f"./datasets/{dataset_name}{'_nopast' if 'removepast' in strategy_to_apply else ''}", "3_classes")

print(f'dataset_name = {dataset_name}')
print(f'model_checkpoint = {model_checkpoint}')
print(f'model_revision = {model_revision}')
print(f'strategy_to_apply = {strategy_to_apply}')

strategy_to_apply_radix = "withpast_" + strategy_to_apply.split("_", 1)[1]

apply_strategy = importlib.import_module(f"contradiction_checking.{strategy_to_apply_radix}").apply_strategy

# train_df = apply_strategy(dataset["train"].to_pandas(), model_checkpoint, model_revision)
# print("Strategy applied on training set")
# validation_df = apply_strategy(dataset["validation"].to_pandas(), model_checkpoint, model_revision)
# print("Strategy applied on validation set")
# test_df = apply_strategy(dataset["test"].to_pandas(), model_checkpoint, model_revision)
# print("Strategy applied on test set")
#
# classifier = LogisticRegression().fit(train_df[["nb_entailed_pairs", "share_entailed_pairs", "nb_contradictory_pairs",
#                                                 "share_contradictory_pairs", "nb_neutral_pairs", "share_neutral_pairs"]
#                                       ].to_numpy(), train_df["label"].to_numpy())
# print("Model trained!")
#
# print("On validation set:", classifier.score(validation_df[["nb_entailed_pairs", "share_entailed_pairs",
#                                                             "nb_contradictory_pairs", "share_contradictory_pairs",
#                                                             "nb_neutral_pairs", "share_neutral_pairs"]
#                                              ].to_numpy(), validation_df["label"].to_numpy()))
# print("On test set:", classifier.score(test_df[["nb_entailed_pairs", "share_entailed_pairs", "nb_contradictory_pairs",
#                                                 "share_contradictory_pairs", "nb_neutral_pairs", "share_neutral_pairs"]
#                                        ].to_numpy(), test_df["label"].to_numpy()))
#
# joblib.dump(classifier, f"./results/joblib_dumps/{dataset_name}/classifier_{model_checkpoint.split('/')[-1]}_{strategy_to_apply}.joblib")

test_df = apply_strategy(dataset["test"].to_pandas(), model_checkpoint, model_revision)


# classifier = LogisticRegression().fit(test_df[["nb_entailed_pairs", "share_entailed_pairs", "nb_contradictory_pairs",
#                                                "share_contradictory_pairs", "nb_neutral_pairs", "share_neutral_pairs"]].to_numpy(), test_df["label"].to_numpy())
# if not os.path.exists(f"./results/joblib_dumps/{dataset_name}/{model_name}"):
#     os.makedirs(f"./results/joblib_dumps/{dataset_name}/{model_name}", exist_ok=True)
#
# joblib.dump(classifier, f"./results/joblib_dumps/{dataset_name}/{model_name}/{strategy_to_apply}.joblib")

f1_metric = load_metric("f1")

# test_df["label"] = test_df["label"].apply(lambda label: 0 if label == 2 else label)

if not os.path.exists(f"./results/threshold/{dataset_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}"):
    os.makedirs(f"./results/threshold/{dataset_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}", exist_ok=True)

with open(f"./results/threshold/{dataset_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}/{strategy_to_apply}.log", "w", encoding="utf8") as file:
    contradiction_threshold, entailment_threshold, max_f1 = maximize_f1_score(test_df["share_contradictory_pairs"], test_df["share_entailed_pairs"], test_df["label"])

    predictions = test_df.apply(lambda row: define_label(row["share_contradictory_pairs"], row["share_entailed_pairs"], contradiction_threshold, entailment_threshold), axis=1).tolist()
    labels = test_df["label"].tolist()

    file.write(f"With contradiction_threshold = {contradiction_threshold}")
    file.write(f"With entailment_threshold = {entailment_threshold}")
    file.write("\nF1 micro: ")
    file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]))
    file.write("\nF1 macro: ")
    file.write(str(max_f1))
    file.write("\n")

print(f"Computing achieved for model '{model_name}' with strategy '{strategy_to_apply}'")
