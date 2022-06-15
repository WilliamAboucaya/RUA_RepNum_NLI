import importlib
import sys

import joblib
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

if len(sys.argv) >= 5:
    dataset_name = sys.argv[1]
    model_checkpoint = sys.argv[2]
    model_revision = sys.argv[3]
    strategy_to_apply = sys.argv[4]
else:
    dataset_name = "rua_nli"
    model_checkpoint = "waboucay/camembert-base-finetuned-nli-repnum_wl-rua_wl"
    model_revision = "main"
    strategy_to_apply = "withpast_sentencecouple_contradictionshare"

dataset = load_dataset(dataset_name, "2_classes")

apply_strategy = importlib.import_module(f"contradiction_checking.{strategy_to_apply}").apply_strategy

train_df = apply_strategy(dataset["train"].to_pandas(), model_checkpoint, model_revision)
validation_df = apply_strategy(dataset["validation"].to_pandas(), model_checkpoint, model_revision)
test_df = apply_strategy(dataset["test"].to_pandas(), model_checkpoint, model_revision)

classifier = LogisticRegression().fit(train_df["share_contradictory_pairs", "nb_contradictory_pairs"], train_df["label"])

print("On validation set:", classifier.score(validation_df["share_contradictory_pairs", "nb_contradictory_pairs"], validation_df["label"]))
print("On test set:", classifier.score(test_df["share_contradictory_pairs", "nb_contradictory_pairs"], test_df["label"]))

joblib.dump(classifier, f"../results/joblib_dumps/{dataset_name}/classifier_{model_checkpoint.split('/')[-1]}_{strategy_to_apply}.joblib")
